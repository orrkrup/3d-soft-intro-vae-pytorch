import argparse
import json
import logging
import random
from datetime import datetime
from os.path import join, exists
import time
import numpy as np

import matplotlib.pyplot as plt
import torch

import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader

import wandb

from soft_intro_vae.utils.util import find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging
from soft_intro_vae.metrics.jsd import jsd_between_point_cloud_sets
from soft_intro_vae.datasets.transforms3d import RandomRotateAxisAngle, RandomTranslate

from soft_intro_vae.models.vae import SoftIntroVAE, reparameterize, ConditionalSoftIntroVAE
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
cudnn.benchmark = True


def calc_jsd_valid(model, dataset, config, prior_std=1.0, split='valid', conditional=False, batches=3):
    st = time.time()
    model.eval()
    device = cuda_setup(config['cuda'], config['gpu'])

    pre_data_split = dataset.split
    dataset.split = split
    num_samples = len(dataset)
    data_loader = DataLoader(dataset, batch_size=int(num_samples / batches),
                             shuffle=False, num_workers=4,
                             drop_last=False, pin_memory=True)
    # We take 3 times as many samples as there are in test data in order to
    # perform JSD calculation in the same manner as in the reference publication
    # noise = torch.FloatTensor(3 * num_samples, train_config['z_size'], 1)
    # noise = prior_std * torch.randn(3 * num_samples, model.zdim)
    # noise = noise.to(device)

    if conditional:
        x_p, x, _ = next(iter(data_loader))
        if x_p.shape[-1] == 3:
            x_p.transpose_(-2, -1)
        x_p = x_p.to(device)
        if x_p.shape[-1] == 3:
            x_p.transpose_(-2, -1)
    else:
        x, _ = next(iter(data_loader))

    x = x.to(device)

    ct = time.time()
    print("jsd prep time: " + str(ct - st))
    # We average JSD computation from 3 independet trials.
    js_results = []
    for _ in range(3):

        if conditional:
            with torch.no_grad():
                x_g = model.sample(x_p)
        else:
            noise = prior_std * torch.randn(3 * num_samples, model.zdim)
            noise = noise.to(device)

            with torch.no_grad():
                x_g = model.decode(noise)

        if x_g.shape[-2:] == (3, 2048):
            x_g.transpose_(-2, -1)
        
        st = time.time()
        print("jsd sample time: " + str(st - ct))
        jsd = jsd_between_point_cloud_sets(x, x_g, voxels=28)
        ct = time.time()
        print("jsd calc time: " + str(ct - st))

        js_results.append(jsd)
    js_result = np.mean(js_results)
    dataset.split = pre_data_split
    return js_result


def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce='sum'):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :return: kld
    """
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    kl = -0.5 * (1 + logvar - logvar_o - logvar.exp() / torch.exp(logvar_o) - (mu - mu_o).pow(2) / torch.exp(
        logvar_o)).sum(1)
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


def apply_transforms(x, transforms):
    if transforms is None:
        return x
    
    if not isinstance(x, list):
        x = [x]

    t = None
    for transform in transforms:
        if 'rotate' == transform:
            new_t = RandomRotateAxisAngle(bsz=x[0].shape[0], axis="Z", device=x[0].device)
        elif 'translate' == transform:
            new_t = RandomTranslate(bsz=x[0].shape[0], device=x[0].device)
        else:
            raise ValueError(f'Invalid transform: {transform}')
        
        if t is None:
            t = new_t
        else:
            t = t.compose(new_t)
    
    pcs = [t.transform_points(pc) for pc in x]
    return pcs[0] if len(pcs) == 1 else pcs


def main(config):
    if config['seed'] >= 0:
        random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        print("random seed: ", config['seed'])

    results_dir = prepare_results_dir(config)
    starting_epoch = find_latest_epoch(results_dir) + 1

    if config['wandb_root']:
        wandb.init(project='3dsintrovae')
        wandb.config.update(config)

    if not exists(join(results_dir, 'config.json')):
        with open(join(results_dir, 'config.json'), mode='w') as f:
            json.dump(config, f)

    setup_logging(results_dir, log_level=logging.INFO)
    log = logging.getLogger(__name__)
    logging.getLogger('matplotlib.font_manager').disabled = True

    device = cuda_setup(config['cuda'], config['gpu'])
    log.debug(f'Device variable: {device}')
    if device.type == 'cuda':
        log.debug(f'Current CUDA device: {torch.cuda.current_device()}')

    weights_path = join(results_dir, 'weights')

    # load dataset
    print("Loading dataset...")
    dataset_name = config['dataset'].lower()
    assert dataset_name == 'shapenet', "Only shapenet dataset supported for this version of the code"
    from soft_intro_vae.datasets.shapenet import ShapeNetDataset
    dataset = ShapeNetDataset(root_dir=config['data_dir'], classes=config['classes'], conditional=config['conditional'])
    assert not config['partial'], "Partial only model not supported with ShapeNet Dataset!"

    log.debug("Selected {} classes. Loaded {} samples.".format(
        'all' if not config['classes'] else ','.join(config['classes']),
        len(dataset)))

    print("Dataset Loaded. ")
    points_dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                   shuffle=config['shuffle'],
                                   num_workers=config['num_workers'],
                                   drop_last=True, pin_memory=True)
    scale = 1 / (3 * config['n_points'])
    conditional = config["conditional"]
    # hyper-parameters
    valid_frequency = config["valid_frequency"]
    num_vae = config["num_vae"]
    beta_rec = config["beta_rec"]
    beta_kl = config["beta_kl"]
    beta_neg = config["beta_neg"]
    gamma_r = config["gamma_r"]

    # model
    if conditional:
        if config["prior_model"]:
            with open(join(config["prior_model"], 'config.json')) as f:
                old_config = json.load(f)
            prior_full_model = SoftIntroVAE(old_config).to(device)
            # TODO: fix magic 02000 number
            prior_full_model.load_state_dict(torch.load(join(config["prior_model"], 'weights', '02000.pth')))
            prior_model = prior_full_model.encoder
            prior_model.eval()
        else:
            prior_model = None
        model = ConditionalSoftIntroVAE(config, prior_model).to(device)
    else:
        model = SoftIntroVAE(config).to(device)

    if config['reconstruction_loss'].lower() == 'chamfer':
        from soft_intro_vae.losses.champfer_loss import ChamferLoss
        reconstruction_loss = ChamferLoss().to(device)
    elif config['reconstruction_loss'].lower() == 'earth_mover':
        from geomloss import SamplesLoss
        reconstruction_loss = SamplesLoss(loss='sinkhorn', debias=False, p=1, blur=1e-3, scaling=0.6)

    else:
        raise ValueError(f'Invalid reconstruction loss. Accepted `chamfer` or '
                         f'`earth_mover`, got: {config["reconstruction_loss"]}')

    if not conditional:
        prior_std = config["prior_std"]
        prior_logvar = np.log(prior_std ** 2)
        prior_mu = 0.0
        print(f'prior: N(0, {prior_std ** 2:.3f})')
    else:
        print(f'prior: conditional')

    # optimizers
    optimizer_e = getattr(optim, config['optimizer']['E']['type'])
    optimizer_e = optimizer_e(model.encoder.parameters(), **config['optimizer']['E']['hyperparams'])
    optimizer_d = getattr(optim, config['optimizer']['D']['type'])
    optimizer_d = optimizer_d(model.decoder.parameters(), **config['optimizer']['D']['hyperparams'])

    scheduler_e = optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=[350, 450, 550], gamma=0.5)
    scheduler_d = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=[350, 450, 550], gamma=0.5)

    if starting_epoch > 1:
        model.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}.pth')))

        optimizer_e.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_optim_e.pth')))
        optimizer_d.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_optim_d.pth')))

    best_res = {"epoch": 0, "jsd": None}

    for epoch in range(starting_epoch, config['max_epochs'] + 1):
        start_epoch_time = datetime.now()

        model.train()

        if epoch <= num_vae:
            bkl = 0
            bre = 0
            nb = 0
            pbar = tqdm(iterable=points_dataloader)
            for i, point_data in enumerate(pbar, 1):
                if conditional:
                    # x_p is the condition, x is the data point
                    x_p, x, _ = point_data
                    x_p = x_p.to(device)

                else:
                    x, _ = point_data

                x = x.to(device)
                
                # Apply transforms, if any
                if conditional:
                    x, x_p = apply_transforms([x, x_p], config.get('transforms', None))
                else:
                    x = apply_transforms(x, config.get('transforms', None))

                # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
                if x.size(-1) == 3:
                    x.transpose_(x.dim() - 2, x.dim() - 1)
                if conditional and x_p.size(-1) == 3:
                    x_p.transpose_(x_p.dim() - 2, x_p.dim() - 1)

                if conditional:
                    x_rec, mu, logvar, prior_mu, prior_logvar = model(x, x_p)
                else:
                    x_rec, mu, logvar = model(x)
                loss_rec = reconstruction_loss(x.permute(0, 2, 1) + 0.5, x_rec.permute(0, 2, 1) + 0.5)

                while len(loss_rec.shape) > 1:
                    loss_rec = loss_rec.sum(-1)
                loss_rec = loss_rec.mean()
                loss_kl = calc_kl(logvar, mu, logvar_o=prior_logvar, mu_o=prior_mu, reduce="mean")
                loss = beta_rec * loss_rec + beta_kl * loss_kl

                bkl += loss_kl
                bre += loss_rec
                nb += 1

                optimizer_e.zero_grad()
                optimizer_d.zero_grad()
                loss.backward()
                optimizer_e.step()
                optimizer_d.step()

                pbar.set_description_str('epoch #{}'.format(epoch))

            # Summarize epoch
            if config['wandb_root']:
                wandb.log({'train/kl': bkl / nb, 'train/rec_err': bre / nb}, step=epoch)

        else:
            batch_kls_real = []
            batch_kls_fake = []
            batch_kls_rec = []
            batch_rec_errs = []
            batch_exp_elbo_f = []
            batch_exp_elbo_r = []
            batch_diff_kls = []
            pbar = tqdm(iterable=points_dataloader)
            for i, point_data in enumerate(pbar, 1):

                if conditional:
                    x_p, x, _ = point_data
                    x_p = x_p.to(device)
                else:
                    x, _ = point_data

                x = x.to(device)

                # Apply transforms, if any
                if conditional:
                    x, x_p = apply_transforms([x, x_p], config.get('transforms', None))
                else:
                    x = apply_transforms(x, config.get('transforms', None))

                # change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
                if x.size(-1) == 3:
                    x.transpose_(x.dim() - 2, x.dim() - 1)
                if conditional and x_p.size(-1) == 3:
                    x_p.transpose_(x_p.dim() - 2, x_p.dim() - 1)

                if not conditional:
                    noise_batch = prior_std * torch.randn(size=(config['batch_size'], model.zdim)).to(device)

                # ----- update E ----- #
                for param in model.encoder.parameters():
                    param.requires_grad = True
                for param in model.decoder.parameters():
                    param.requires_grad = False

                if conditional:
                    fake = model.sample(x_p)

                    real_mu, real_logvar, prior_mu, prior_logvar = model.encode(x, x_p)
                    z = reparameterize(real_mu, real_logvar)
                    x_rec = model.decoder(z, condition=prior_mu.detach())
                else:
                    fake = model.sample(noise_batch)

                    real_mu, real_logvar = model.encode(x)
                    z = reparameterize(real_mu, real_logvar)
                    x_rec = model.decoder(z)

                loss_rec = reconstruction_loss(x.permute(0, 2, 1) + 0.5, x_rec.permute(0, 2, 1) + 0.5)

                while len(loss_rec.shape) > 1:
                    loss_rec = loss_rec.sum(-1)
                loss_rec = loss_rec.mean()

                loss_real_kl = calc_kl(real_logvar, real_mu, mu_o=prior_mu, logvar_o=prior_logvar, reduce="mean")
                if conditional:
                    rec_rec, rec_mu, rec_logvar, _, _ = model(x_rec.detach(), x_p.detach())
                    rec_fake, fake_mu, fake_logvar, _, _ = model(fake.detach(), x_p.detach())
                else:
                    rec_rec, rec_mu, rec_logvar = model(x_rec.detach())
                    rec_fake, fake_mu, fake_logvar = model(fake.detach())

                kl_rec = calc_kl(rec_logvar, rec_mu, logvar_o=prior_logvar, mu_o=prior_mu, reduce="none")
                kl_fake = calc_kl(fake_logvar, fake_mu, logvar_o=prior_logvar, mu_o=prior_mu, reduce="none")

                loss_rec_rec_e = reconstruction_loss(x_rec.detach().permute(0, 2, 1) + 0.5,
                                                     rec_rec.permute(0, 2, 1) + 0.5)

                while len(loss_rec_rec_e.shape) > 1:
                    loss_rec_rec_e = loss_rec_rec_e.sum(-1)
                loss_rec_fake_e = reconstruction_loss(fake.permute(0, 2, 1) + 0.5, rec_fake.permute(0, 2, 1) + 0.5)

                while len(loss_rec_fake_e.shape) > 1:
                    loss_rec_fake_e = loss_rec_fake_e.sum(-1)

                expelbo_rec = (-2 * scale * (beta_rec * loss_rec_rec_e + beta_neg * kl_rec)).exp().mean()
                expelbo_fake = (-2 * scale * (beta_rec * loss_rec_fake_e + beta_neg * kl_fake)).exp().mean()

                loss_margin = scale * beta_kl * loss_real_kl + 0.25 * (expelbo_rec + expelbo_fake)

                lossE = scale * beta_rec * loss_rec + loss_margin
                optimizer_e.zero_grad()
                lossE.backward()
                optimizer_e.step()

                # ----- update D ----- #
                for param in model.encoder.parameters():
                    param.requires_grad = False
                for param in model.decoder.parameters():
                    param.requires_grad = True

                with torch.no_grad():
                    z = reparameterize(real_mu.detach(), real_logvar.detach())

                if conditional:
                    fake = model.sample(x_p)
                    rec = model.decoder(z.detach(), condition=prior_mu.detach())
                else:
                    fake = model.sample(noise_batch)
                    rec = model.decoder(z.detach())
                loss_rec = reconstruction_loss(x.permute(0, 2, 1) + 0.5, rec.permute(0, 2, 1) + 0.5)

                while len(loss_rec.shape) > 1:
                    loss_rec = loss_rec.sum(-1)
                loss_rec = loss_rec.mean()

                if conditional:
                    rec_mu, rec_logvar, _, _ = model.encode(rec, x_p)
                    fake_mu, fake_logvar, _, _ = model.encode(fake, x_p)
                else:
                    rec_mu, rec_logvar = model.encode(rec)
                    fake_mu, fake_logvar = model.encode(fake)

                z_rec = reparameterize(rec_mu, rec_logvar)
                z_fake = reparameterize(fake_mu, fake_logvar)

                if conditional:
                    rec_rec = model.decode(z_rec.detach(), x_p.detach())
                    rec_fake = model.decode(z_fake.detach(), x_p.detach())
                else:
                    rec_rec = model.decode(z_rec.detach())
                    rec_fake = model.decode(z_fake.detach())

                loss_rec_rec = reconstruction_loss(rec.detach().permute(0, 2, 1) + 0.5, rec_rec.permute(0, 2, 1) + 0.5)

                while len(loss_rec_rec.shape) > 1:
                    loss_rec_rec = loss_rec.sum(-1)
                loss_rec_rec = loss_rec_rec.mean()
                loss_fake_rec = reconstruction_loss(fake.detach().permute(0, 2, 1) + 0.5,
                                                    rec_fake.permute(0, 2, 1) + 0.5)

                while len(loss_fake_rec.shape) > 1:
                    loss_fake_rec = loss_rec.sum(-1)
                loss_fake_rec = loss_fake_rec.mean()

                lossD_rec_kl = calc_kl(rec_logvar, rec_mu, logvar_o=prior_logvar, mu_o=prior_mu, reduce="mean")
                lossD_fake_kl = calc_kl(fake_logvar, fake_mu, logvar_o=prior_logvar, mu_o=prior_mu, reduce="mean")

                lossD = scale * (loss_rec * beta_rec + (
                        lossD_rec_kl + lossD_fake_kl) * 0.5 * beta_kl + gamma_r * 0.5 * beta_rec * (
                                         loss_rec_rec + loss_fake_rec))

                optimizer_d.zero_grad()
                lossD.backward()
                optimizer_d.step()

                # scheduler_e.step()
                # scheduler_d.step()
                if torch.isnan(lossD):
                    raise SystemError("loss is Nan")

                diff_kl = -loss_real_kl.data.cpu() + lossD_fake_kl.data.cpu()
                batch_diff_kls.append(diff_kl)
                batch_kls_real.append(loss_real_kl.data.cpu().item())
                batch_kls_fake.append(lossD_fake_kl.cpu().item())
                batch_kls_rec.append(lossD_rec_kl.data.cpu().item())
                batch_rec_errs.append(loss_rec.data.cpu().item())
                batch_exp_elbo_f.append(expelbo_fake.data.cpu())
                batch_exp_elbo_r.append(expelbo_rec.data.cpu())

                pbar.set_description_str('epoch #{}'.format(epoch))
                pbar.set_postfix(r_loss=loss_rec.data.cpu().item(), kl=loss_real_kl.data.cpu().item(),
                                 diff_kl=diff_kl.item(), expelbo_f=expelbo_fake.cpu().item())

            # Summarize epoch
            kl_real = np.mean(batch_kls_real)
            kl_fake = np.mean(batch_kls_fake)
            kl_rec = np.mean(batch_kls_rec)
            rec_err = np.mean(batch_rec_errs)
            exp_elbo_f = np.mean(batch_exp_elbo_f)
            exp_elbo_r = np.mean(batch_exp_elbo_r)
            diff_kl = np.mean(batch_diff_kls)
            # epoch summary
            print('#' * 50)
            print(f'Epoch {epoch} Summary:')
            print(f'beta_rec: {beta_rec}, beta_kl: {beta_kl}, beta_neg: {beta_neg}')
            print(
                f'rec: {rec_err:.3f}, kl: {kl_real:.3f}, kl_fake: {kl_fake:.3f}, kl_rec: {kl_rec:.3f}')
            print(
                f'diff_kl: {diff_kl:.3f}, exp_elbo_f: {exp_elbo_f:.4e}, exp_elbo_r: {exp_elbo_r:.4e}')
            if best_res['jsd'] is not None:
                print(f'best jsd: {best_res["jsd"]}, epoch: {best_res["epoch"]}')
            print(f'time: {datetime.now() - start_epoch_time}')
            print('#' * 50)

            if config['wandb_root']:
                wandb.log({'train/kl_real': kl_real,
                           'train/kl_fake': kl_fake,
                           'train/kl_rec': kl_rec,
                           'train/rec_err': rec_err,
                           'train/exp_elbo_fake': exp_elbo_f,
                           'train/exp_elbo_real': exp_elbo_r,
                           'train/diff_kl': diff_kl}, step=epoch)

        pbar.close()
        scheduler_e.step()
        scheduler_d.step()

        # save intermediate results
        model.eval()
        sample_size = 5
        with torch.no_grad():
            p_data_loader = DataLoader(dataset, batch_size=sample_size,
                                       shuffle=True, num_workers=4,
                                       drop_last=False, pin_memory=True)
            if conditional:
                prior_batch, x_batch, _ = next(iter(p_data_loader))
                x_batch = x_batch.to(device)
                prior_batch = prior_batch.to(device)
                
                x_batch, prior_batch = apply_transforms([x_batch, prior_batch], config.get('transforms', None))

                if x_batch.size(-1) == 3:
                    x_batch.transpose_(x_batch.dim() - 2, x_batch.dim() - 1)
                
                if conditional and prior_batch.size(-1) == 3:
                    prior_batch.transpose_(prior_batch.dim() - 2, prior_batch.dim() - 1)

                x_rec, _, _, _, _ = model(x_batch, prior_batch, deterministic=True)
                fake = model.sample(torch.stack(prior_batch.size(0) * [prior_batch[0]])).data.cpu().numpy()
                # fake[0] = prior_batch[0].cpu().numpy()
            else:
                x_batch, _ = next(iter(p_data_loader))
                x_batch = x_batch.to(device)

                x_batch = apply_transforms(x_batch, config.get('transforms', None))
                noise_batch = prior_std * torch.randn(size=(sample_size, model.zdim)).to(device)

                if x_batch.size(-1) == 3:
                    x_batch.transpose_(x_batch.dim() - 2, x_batch.dim() - 1)
                
                x_rec, _, _ = model(x_batch, deterministic=True)
                fake = model.sample(noise_batch).data.cpu().numpy()

            x_rec = x_rec.data.cpu().numpy()

            fig = plt.figure(dpi=350)
            pc_batches = [x_batch.data.cpu().numpy(), x_rec, fake]
            if conditional:
                pc_batches.insert(1, prior_batch.data.cpu().numpy())

        if config['wandb_root']:
            titles = ['input', 'reconstruction', 'random_sample']
            if conditional:
                titles.insert(1, 'prior (partial)')

                def add_color(pc, color):
                    return np.concatenate((pc, np.tile([[c] for c in color], (1, pc.shape[-1]))), axis=-2)
                overlay = add_color(pc_batches[1][0], [223, 116, 62])
                pc_batches[-1] = np.stack([np.concatenate((add_color(pc, [100, 100, 100]), overlay), axis=-1) for pc in pc_batches[-1]], axis=0)

            for t, src in zip(titles, pc_batches):
                mesh_samples = src.transpose(0, 2, 1)
                wandb.log({t: [wandb.Object3D({'type': 'lidar/beta',
                                               'points': smp,
                                               'vectors': np.array([{'start': [0., 0., 0.], 'end': [0., 0., 0.1], "color": [0, 0, 255]},
                                                                    {'start': [0., 0., 0.], 'end': [0., 0.1, 0.], "color": [0, 255, 0]},
                                                                    {'start': [0., 0., 0.], 'end': [0.1, 0., 0.], "color": [255, 0, 0]}])
                                               }) for smp in mesh_samples]}, step=epoch)

        # JSD Validation
        if epoch % valid_frequency == 0:
            print("calculating valid jsd... starting time: " + time.strftime("%H:%M:%S", time.localtime()))
            model.eval()
            with torch.no_grad():
                jsd = calc_jsd_valid(model, dataset, config, prior_std=(None if conditional else prior_std),
                                     conditional=conditional, batches=(3 if 'drinks' == dataset_name else 1),
                                     split=('val' if '3depn' == dataset_name else 'valid'))
            print(f'epoch: {epoch}, jsd: {jsd:.4f}, end time: ' + time.strftime("%H:%M:%S", time.localtime()))
            if best_res['jsd'] is None:
                best_res['jsd'] = jsd
                best_res['epoch'] = epoch
            elif best_res['jsd'] > jsd:
                print(f'epoch: {epoch}: best jsd updated: {best_res["jsd"]} -> {jsd}')
                best_res['jsd'] = jsd
                best_res['epoch'] = epoch
                # save
                torch.save(model.state_dict(), join(weights_path, f'{epoch:05}_jsd_{jsd:.4f}.pth'))

            if config['wandb_root']:
                wandb.log({'validation_jsd': jsd}, step=epoch)
            # if config['tb_root']:
            #     tb_writer.add_scalar('validation_jsd', jsd, global_step=epoch)

        if epoch % config['save_frequency'] == 0:
            torch.save(model.state_dict(), join(weights_path, f'{epoch:05}.pth'))
            torch.save(optimizer_e.state_dict(),
                       join(weights_path, f'{epoch:05}_optim_e.pth'))
            torch.save(optimizer_d.state_dict(),
                       join(weights_path, f'{epoch:05}_optim_d.pth'))


def get_config(config_filename):
    config = None
    if config_filename is not None and config_filename.endswith('.json'):
        with open(config_filename) as f:
            config = json.load(f)

    return config


if __name__ == '__main__':
    # logger = logging.getLogger(__name__)
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path')
    args = parser.parse_args()

    # args.config = './settings/soft_intro_vae_hp.json'
    config = get_config(args.config)

    assert config is not None

    main(config)
