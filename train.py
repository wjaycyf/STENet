import os
import time
import math
import numbers
import torch

from utils import Train_Report, TestReport, SaveManager


class Trainer:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        if self.config.save_train_img:
            self.save_manager = SaveManager(config)
        self.criterion = torch.nn.L1Loss()

        milestones = [260, 360, 380, 390]
        # optimizer and scheduler for degradation learning network
        self.optimizer_D = torch.optim.Adam(self.model.degradation_learning_network.parameters(), lr=self.config.lr)
        self.scheduler_D = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_D, milestones=milestones, gamma=0.5, last_epoch=-1)

        # optimizer and scheduler for restoration network
        if self.config.stage == 2:
            self.optimizer_R = torch.optim.Adam(self.model.restoration_network.parameters(), lr=self.config.lr)
            self.scheduler_R = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_R, milestones=milestones, gamma=0.5, last_epoch=-1)

        self.checkpoint_path = os.path.join(self.config.save_dir, f'model_stage{self.config.stage}')
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.model.cuda()
        self.memory_train_mode = getattr(self.config, 'memory_train_mode', 'off')
        self.memory_eval_mode = getattr(self.config, 'memory_eval_mode', 'sequential')
        self.memory_debug = getattr(self.config, 'memory_debug', False)
        self.grad_clip_norm = float(getattr(self.config, 'grad_clip_norm', 1.0))
        self.memory_reset_count = 0
        self._validate_memory_modes()

    def _validate_memory_modes(self):
        valid_train = {'off', 'sequential_epoch'}
        valid_eval = {'off', 'sequential'}
        if self.memory_train_mode not in valid_train:
            raise ValueError(f'unsupported memory_train_mode: {self.memory_train_mode}')
        if self.memory_eval_mode not in valid_eval:
            raise ValueError(f'unsupported memory_eval_mode: {self.memory_eval_mode}')

    def _set_memory_enabled(self, enabled):
        if hasattr(self.model, 'set_memory_enabled'):
            self.model.set_memory_enabled(enabled)

    def _reset_memory(self):
        if hasattr(self.model, 'reset_memory'):
            self.model.reset_memory()
            self.memory_reset_count += 1

    def _memory_stats_str(self):
        if not self.memory_debug or not hasattr(self.model, 'get_memory_stats'):
            return ''
        stats = self.model.get_memory_stats()
        d_stats = stats.get('D', {}) if isinstance(stats, dict) else {}
        gate = d_stats.get('gate_mean', None)
        if gate is None:
            return ''
        gate_val = gate.item() if hasattr(gate, 'item') else float(gate)
        return f'\tMemGate(D): {gate_val:.5f}\tMemReset: {self.memory_reset_count}'

    def _memory_context_str(self, batch_size, scene_token):
        scene = 'None' if scene_token is None else str(scene_token)
        return f'\tBatch: {batch_size}\tScene: {scene}\tMemReset: {self.memory_reset_count}'

    def _ensure_finite(self, name, value):
        if isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value).all().item()):
                raise FloatingPointError(f'non-finite detected in {name}')
            return
        if isinstance(value, numbers.Number):
            if not math.isfinite(float(value)):
                raise FloatingPointError(f'non-finite detected in {name}')
            return

    def _state_dict_is_finite(self, state_dict):
        for value in state_dict.values():
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all().item()):
                return False
        return True

    def _clip_restoration_gradients(self):
        if self.config.stage != 2 or self.grad_clip_norm <= 0:
            return
        if hasattr(self.model.restoration_network, 'kcsr_block'):
            torch.nn.utils.clip_grad_norm_(
                self.model.restoration_network.kcsr_block.parameters(),
                max_norm=self.grad_clip_norm,
            )

    def _ensure_gradients_finite(self, module, module_name):
        for name, param in module.named_parameters():
            if param.grad is None:
                continue
            if not bool(torch.isfinite(param.grad).all().item()):
                raise FloatingPointError(f'non-finite gradient detected in {module_name}.{name}')

    def _can_save_current_model(self):
        d_state = self.model.degradation_learning_network.state_dict()
        if not self._state_dict_is_finite(d_state):
            print('skip checkpoint save: degradation network contains non-finite parameters')
            return False
        if self.config.stage == 2:
            r_state = self.model.restoration_network.state_dict()
            if not self._state_dict_is_finite(r_state):
                print('skip checkpoint save: restoration network contains non-finite parameters')
                return False
        return True

    def save_checkpoint(self, epoch):
        if not self._can_save_current_model():
            return
        D_state_dict = {'epoch': epoch,
                        'model_D_state_dict': self.model.degradation_learning_network.state_dict(),
                        'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                        'scheduler_D_state_dict': self.scheduler_D.state_dict()}
        torch.save(D_state_dict, self.checkpoint_path + '/model_D_latest.pt')
        torch.save(D_state_dict, self.checkpoint_path + '/model_D_' + str(epoch) + '.pt')

        if self.config.stage == 2:
            R_state_dict = {'epoch': epoch,
                            'model_R_state_dict': self.model.restoration_network.state_dict(),
                            'optimizer_R_state_dict': self.optimizer_R.state_dict(),
                            'scheduler_R_state_dict': self.scheduler_R.state_dict()}
            torch.save(R_state_dict, self.checkpoint_path + '/model_R_latest.pt')
            torch.save(R_state_dict, self.checkpoint_path + '/model_R_' + str(epoch) + '.pt')

    def save_best_model(self, epoch):
        if not self._can_save_current_model():
            return
        D_state_dict = {'epoch': epoch,
                        'model_D_state_dict': self.model.degradation_learning_network.state_dict(),
                        'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                        'scheduler_D_state_dict': self.scheduler_D.state_dict()}
        torch.save(D_state_dict, self.checkpoint_path + '/model_D_best.pt')

        if self.config.stage == 2:
            R_state_dict = {'epoch': epoch,
                            'model_R_state_dict': self.model.restoration_network.state_dict(),
                            'optimizer_R_state_dict': self.optimizer_R.state_dict(),
                            'scheduler_R_state_dict': self.scheduler_R.state_dict()}
            torch.save(R_state_dict, self.checkpoint_path + '/model_R_best.pt')

    def load_checkpoint(self, epoch=None):
        if epoch is None:
            if not os.path.exists(self.checkpoint_path + '/model_D_latest.pt'):
                print(f'no latest checkpoint found in {self.checkpoint_path}, starting from epoch 0')
                return 0

            D_state_dict = torch.load(self.checkpoint_path + '/model_D_latest.pt')
            self.model.degradation_learning_network.load_state_dict(D_state_dict['model_D_state_dict'])
            self.optimizer_D.load_state_dict(D_state_dict['optimizer_D_state_dict'])
            self.scheduler_D.load_state_dict(D_state_dict['scheduler_D_state_dict'])
            last_epoch = D_state_dict['epoch']
            print(f'load degradation learning network status from {self.checkpoint_path}/model_D_latest.pt, epoch: {last_epoch}')

            if self.config.stage == 2:
                R_state_dict = torch.load(self.checkpoint_path + '/model_R_latest.pt')
                self.model.restoration_network.load_state_dict(R_state_dict['model_R_state_dict'])
                self.optimizer_R.load_state_dict(R_state_dict['optimizer_R_state_dict'])
                self.scheduler_R.load_state_dict(R_state_dict['scheduler_R_state_dict'])
                last_epoch = R_state_dict['epoch']
                print(f'load restoration network status from {self.checkpoint_path}/model_R_latest.pt, epoch: {last_epoch}')

        else:
            D_state_dict = torch.load(self.checkpoint_path + '/model_D_' + str(epoch) + '.pt')
            self.model.degradation_learning_network.load_state_dict(D_state_dict['model_D_state_dict'])
            self.optimizer_D.load_state_dict(D_state_dict['optimizer_D_state_dict'])
            self.scheduler_D.load_state_dict(D_state_dict['scheduler_D_state_dict'])
            last_epoch = D_state_dict['epoch']
            print(f'load degradation learning network status from {self.checkpoint_path}/model_D_{epoch}.pt, epoch: {last_epoch}')

            if self.config.stage == 2:
                R_state_dict = torch.load(self.checkpoint_path + '/model_R_' + str(epoch) + '.pt')
                self.model.restoration_network.load_state_dict(R_state_dict['model_R_state_dict'])
                self.optimizer_R.load_state_dict(R_state_dict['optimizer_R_state_dict'])
                self.scheduler_R.load_state_dict(R_state_dict['scheduler_R_state_dict'])
                last_epoch = R_state_dict['epoch']
                print(f'load restoration network status from {self.checkpoint_path}/model_R_{epoch}.pt, epoch: {last_epoch}')

        return last_epoch

    def load_best_model(self):
        D_state_dict = torch.load(self.checkpoint_path + '/model_D_best.pt')
        self.model.degradation_learning_network.load_state_dict(D_state_dict['model_D_state_dict'])
        print(f'load degradation learning network status from {self.checkpoint_path}/model_D_best.pt, epoch: {D_state_dict["epoch"]}')

        if self.config.stage == 2:
            R_state_dict = torch.load(self.checkpoint_path + '/model_R_best.pt')
            self.model.restoration_network.load_state_dict(R_state_dict['model_R_state_dict'])
            print(f'load restoration network status from {self.checkpoint_path}/model_R_best.pt, epoch: {R_state_dict["epoch"]}')

    def load_best_stage1_model(self):
        if self.config.stage == 1:
            self.load_stage1_finetune_init()
            return
        if self.config.stage == 2:
            self.load_stage2_finetune_init()
            return
        raise ValueError(f'Unsupported training stage: {self.config.stage}')

    def _load_model_state(self, module, state_dict, strict, tag):
        load_result = module.load_state_dict(state_dict, strict=strict)
        missing = getattr(load_result, 'missing_keys', [])
        unexpected = getattr(load_result, 'unexpected_keys', [])
        if missing or unexpected:
            print(f'[{tag}] strict={strict} missing={len(missing)} unexpected={len(unexpected)}')
            if missing:
                print(f'[{tag}] missing sample: {missing[:5]}')
            if unexpected:
                print(f'[{tag}] unexpected sample: {unexpected[:5]}')

    def load_stage1_finetune_init(self):
        print('stage1 training starts from scratch; no initialization weights are loaded')

    def load_stage2_finetune_init(self):
        stage1_result_best_d = os.path.join(self.config.save_dir, 'model_stage1', 'model_D_best.pt')
        if not os.path.exists(stage1_result_best_d):
            raise FileNotFoundError(
                f'retrained stage1 best D not found: {stage1_result_best_d}. '
                f'Please finish stage1 retraining first.'
            )
        d_state = torch.load(stage1_result_best_d)
        self._load_model_state(
            self.model.degradation_learning_network,
            d_state['model_D_state_dict'],
            strict=True,
            tag='stage2_init_D_from_stage1_results'
        )
        print(f'load stage2 D init from stage1 result best: {stage1_result_best_d}, epoch: {d_state.get("epoch", "N/A")}')
        print('stage2 restoration network keeps random initialization; no stage2 weights are loaded')

    def train(self, dataloader, train_log, global_step):
        self.model.train()
        self.memory_reset_count = 0
        report = Train_Report()
        start = time.time()
        self._set_memory_enabled(self.memory_train_mode == 'sequential_epoch')
        last_scene_token = None
        last_batch_size = None

        for idx, batch in enumerate(dataloader):
            if len(batch) == 5:
                lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow, scene_token = batch
                if isinstance(scene_token, (list, tuple)):
                    scene_token = scene_token[0]
            else:
                lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow = batch
                scene_token = None

            if self.memory_train_mode == 'off':
                self._reset_memory()
            elif self.memory_train_mode == 'sequential_epoch':
                if scene_token != last_scene_token:
                    self._reset_memory()
                last_scene_token = scene_token

            lr_blur_seq = lr_blur_seq.cuda()
            hr_sharp_seq = hr_sharp_seq.cuda()
            lr_sharp_seq = lr_sharp_seq.cuda()
            flow = flow.cuda()
            batch_size = lr_blur_seq.shape[0]
            if self.memory_train_mode == 'sequential_epoch':
                if last_batch_size is not None and batch_size != last_batch_size:
                    self._reset_memory()
                last_batch_size = batch_size

            result_dict = self.model(lr_blur_seq, hr_sharp_seq)
            _, _, t, _, _ = lr_blur_seq.shape

            # pretrain degradation learning network
            if self.config.stage == 1:
                self._ensure_finite("result_dict['recon']", result_dict['recon'])
                recon_loss = self.criterion(result_dict['recon'], lr_blur_seq[:, :, t//2, :, :])
                hr_warping_loss = self.config.hr_warping_loss_weight * self.criterion(result_dict['hr_warp'], hr_sharp_seq[:, :, t//2:t//2+1, :, :].repeat([1,1,t,1,1]))
                # RAFT pseudo-GT optical flow loss
                flow_loss = self.config.flow_loss_weight * self.criterion(result_dict['image_flow'], flow)
                # TA loss for degradation learning network
                D_TA_loss = self.config.D_TA_loss_weight * self.criterion(result_dict['F_sharp_D'], lr_sharp_seq)

                total_loss = recon_loss + hr_warping_loss + flow_loss + D_TA_loss
                self._ensure_finite('stage1_total_loss', total_loss)

                self.optimizer_D.zero_grad()
                total_loss.backward()
                self._ensure_gradients_finite(self.model.degradation_learning_network, 'degradation_learning_network')
                self.optimizer_D.step()

                report.update(batch_size, 0, recon_loss.item(), hr_warping_loss.item(), 0, flow_loss.item(), D_TA_loss.item(), 0, total_loss.item())

            # train full network
            elif self.config.stage == 2:
                self._ensure_finite("result_dict['output']", result_dict['output'])
                restoration_loss = self.criterion(result_dict['output'], hr_sharp_seq[:, :, t//2, :, :])
                recon_loss = self.config.Net_D_weight * self.criterion(result_dict['recon'], lr_blur_seq[:, :, t//2, :, :])
                lr_warping_loss = self.config.lr_warping_loss_weight * self.criterion(result_dict['lr_warp'], lr_blur_seq[:, :, t//2:t//2 + 1, :, :].repeat([1,1,t,1,1]))
                hr_warping_loss = self.config.Net_D_weight * self.config.hr_warping_loss_weight * self.criterion(result_dict['hr_warp'], hr_sharp_seq[:, :, t//2:t//2+1, :, :].repeat([1,1,t,1,1]))
                # RAFT pseudo-GT optical flow loss
                flow_loss = self.config.Net_D_weight * self.config.flow_loss_weight * self.criterion(result_dict['image_flow'], flow)
                # TA loss for degradation learning network and restoration network
                R_TA_loss = self.config.R_TA_loss_weight * self.criterion(result_dict['F_sharp_R'], lr_sharp_seq)
                D_TA_loss = self.config.Net_D_weight * self.config.D_TA_loss_weight * self.criterion(result_dict['F_sharp_D'], lr_sharp_seq)
                
                total_loss = restoration_loss + recon_loss + hr_warping_loss + lr_warping_loss + flow_loss + R_TA_loss + D_TA_loss
                self._ensure_finite('restoration_loss', restoration_loss)
                self._ensure_finite('stage2_total_loss', total_loss)

                self.optimizer_D.zero_grad()
                self.optimizer_R.zero_grad()
                total_loss.backward()
                self._clip_restoration_gradients()
                self._ensure_gradients_finite(self.model.degradation_learning_network, 'degradation_learning_network')
                self._ensure_gradients_finite(self.model.restoration_network, 'restoration_network')
                self.optimizer_D.step()
                self.optimizer_R.step()

                report.update(batch_size, restoration_loss.item(), recon_loss.item(), hr_warping_loss.item(), lr_warping_loss.item(), flow_loss.item(), D_TA_loss.item(), R_TA_loss.item(), total_loss.item())

            global_step += 1

            if global_step % 100 == 0 or idx == len(dataloader) - 1:
                lr_D = self.scheduler_D.optimizer.state_dict()['param_groups'][0]['lr']
                lr_R = self.scheduler_R.optimizer.state_dict()['param_groups'][0]['lr'] if self.config.stage == 2 else None

                period_time = time.time() - start
                prefix_str = f'[{global_step}/{len(dataloader) * self.config.num_epochs}]\t'
                result_str = report.result_str(lr_D, lr_R, period_time)
                result_str += self._memory_context_str(batch_size, scene_token)
                result_str += self._memory_stats_str()

                train_log.write(prefix_str + result_str)
                start = time.time()
                report.__init__()

                if self.config.save_train_img:
                    if self.config.stage == 1:
                        src = [lr_blur_seq[:, :, t // 2, :, :], result_dict['recon']]
                    elif self.config.stage == 2:
                        src = [lr_blur_seq[:, :, t // 2, :, :], result_dict['recon'], result_dict['output'], hr_sharp_seq[:, :, t // 2, :, :]]
                    self.save_manager.save_batch_images(src, batch_size, global_step)

        self.scheduler_D.step()
        if self.config.stage == 2:
            self.scheduler_R.step()

        return global_step

    def validate(self, dataloader, val_log, epoch):
        self.model.eval()
        self.memory_reset_count = 0
        self._set_memory_enabled(self.memory_eval_mode == 'sequential')
        self._reset_memory()
        report = Train_Report()
        start = time.time()
        last_scene_token = None

        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                if len(batch) == 5:
                    lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow, scene_token = batch
                    if isinstance(scene_token, (list, tuple)):
                        scene_token = scene_token[0]
                else:
                    lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow = batch
                    scene_token = None

                if self.memory_eval_mode == 'sequential':
                    if scene_token is not None and scene_token != last_scene_token:
                        self._reset_memory()
                    last_scene_token = scene_token

                lr_blur_seq = lr_blur_seq.cuda()
                hr_sharp_seq = hr_sharp_seq.cuda()
                lr_sharp_seq = lr_sharp_seq.cuda()
                flow = flow.cuda()

                result_dict = self.model(lr_blur_seq, hr_sharp_seq)

                batch_size, _, t, _, _ = lr_blur_seq.shape

                if self.config.stage == 1:
                    self._ensure_finite("validate_result_dict['recon']", result_dict['recon'])
                    recon_loss = self.criterion(result_dict['recon'], lr_blur_seq[:, :, t // 2, :, :])
                    hr_warping_loss = self.config.hr_warping_loss_weight * self.criterion(result_dict['hr_warp'], hr_sharp_seq[:, :, t // 2:t // 2 + 1, :, :].repeat([1, 1, t, 1, 1]))
                    flow_loss = self.config.flow_loss_weight * self.criterion(result_dict['image_flow'], flow)
                    D_TA_loss = self.config.D_TA_loss_weight * self.criterion(result_dict['F_sharp_D'], lr_sharp_seq)

                    total_loss = recon_loss + hr_warping_loss + flow_loss + D_TA_loss
                    report.update(batch_size, 0, recon_loss.item(), hr_warping_loss.item(), 0, flow_loss.item(), D_TA_loss.item(), 0, total_loss.item())
                    report.update_recon_metric(result_dict['recon'], lr_blur_seq[:, :, t // 2, :, :])

                elif self.config.stage == 2:
                    self._ensure_finite("validate_result_dict['output']", result_dict['output'])
                    restoration_loss = self.criterion(result_dict['output'], hr_sharp_seq[:, :, t // 2, :, :])
                    recon_loss = self.config.Net_D_weight * self.criterion(result_dict['recon'], lr_blur_seq[:, :, t // 2, :, :])
                    lr_warping_loss = self.config.lr_warping_loss_weight * self.criterion(result_dict['lr_warp'], lr_blur_seq[:, :, t // 2:t // 2 + 1, :, :].repeat([1, 1, t, 1, 1]))
                    hr_warping_loss = self.config.Net_D_weight * self.config.hr_warping_loss_weight * self.criterion(result_dict['hr_warp'], hr_sharp_seq[:, :, t // 2:t // 2 + 1, :, :].repeat([1, 1, t, 1, 1]))
                    flow_loss = self.config.Net_D_weight * self.config.flow_loss_weight * self.criterion(result_dict['image_flow'], flow)
                    R_TA_loss = self.config.R_TA_loss_weight * self.criterion(result_dict['F_sharp_R'], lr_sharp_seq)
                    D_TA_loss = self.config.Net_D_weight * self.config.D_TA_loss_weight * self.criterion(result_dict['F_sharp_D'], lr_sharp_seq)

                    total_loss = restoration_loss + recon_loss + hr_warping_loss + lr_warping_loss + flow_loss + R_TA_loss + D_TA_loss
                    report.update(batch_size, restoration_loss.item(), recon_loss.item(), hr_warping_loss.item(), lr_warping_loss.item(), flow_loss.item(), D_TA_loss.item(), R_TA_loss.item(), total_loss.item())
                    report.update_recon_metric(result_dict['recon'], lr_blur_seq[:, :, t // 2, :, :])
                    report.update_recon_metric(result_dict['output'], hr_sharp_seq[:, :, t//2, :, :])

        period_time = time.time() - start
        prefix_str = f'[{epoch}/{self.config.num_epochs}]\t'
        result_str = report.val_result_str(period_time)
        result_str += self._memory_stats_str()

        val_log.write(prefix_str + result_str)

        if self.config.stage == 1:
            return report.recon_psnr
        elif self.config.stage == 2:
            return report.psnr

    def test(self, dataloader):
        from utils import denorm
        self.model.eval()
        self._set_memory_enabled(self.memory_eval_mode == 'sequential')
        self._reset_memory()
        last_scene_token = None

        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                if len(batch) == 3:
                    lr_blur_seq, filename, scene_token = batch
                    if isinstance(scene_token, (list, tuple)):
                        scene_token = scene_token[0]
                else:
                    lr_blur_seq, filename = batch
                    scene_token = None

                if self.memory_eval_mode == 'sequential':
                    if scene_token is not None and scene_token != last_scene_token:
                        self._reset_memory()
                    last_scene_token = scene_token

                lr_blur_seq = lr_blur_seq.cuda()

                result_dict = self.model(lr_blur_seq)
                output = result_dict['output']

                output = output.squeeze(dim=0)
                output = denorm(output)

                filename = filename[0]
                filepath = os.path.basename(os.path.dirname(filename))
                filename = os.path.basename(filename)
                filename = os.path.join(self.config.save_dir, 'test', filepath, filename)
                self.save_manager.save_image(output, filename)

    def test_quantitative_result(self, gt_dir, output_dir, image_border):
        import cv2
        import glob

        report = TestReport(output_dir)
        scene_list = sorted(glob.glob(os.path.join(gt_dir, '*')))

        for scene in scene_list:
            scene_name = os.path.basename(scene)
            filelist = sorted(glob.glob(os.path.join(scene, '*.png')))
            report.scene_init(scene_name)
            for filename in filelist[image_border:-image_border]:
                gt_img = cv2.imread(filename)
                output_img = cv2.imread(os.path.join(output_dir, scene_name, os.path.basename(filename)))
                report.update_metric(gt_img, output_img, os.path.basename(filename))
            report.scene_del(scene_name)
