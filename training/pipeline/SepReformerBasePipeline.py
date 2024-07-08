from .utils import TrainPipeline, get_all_cuda_device_ids,check_device_is_cuda
from torch.nn import Module
from ..optimizer import BASE_OPTIMIZER,BASE_OPTIMIZER_CONFIG
from ..scheduler import WarmupConstantSchedule,WarmupConstantScheduleConfig,ReduceLROnPlateauBaseConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..loss import *
import time
from tqdm import tqdm
import gc

class SepReformerBasePipeLine(TrainPipeline):
    def __init__(
            self, 
            model:Module,
            train_dataloader,
            validate_dataloader, 
            optimizer = BASE_OPTIMIZER,
            optimizer_config: dict = BASE_OPTIMIZER_CONFIG,
            main_scheduler = ReduceLROnPlateau,
            main_scheduler_config = ReduceLROnPlateauBaseConfig,
            warm_up_scheduler = WarmupConstantSchedule,
            warm_up_scheduler_config = WarmupConstantScheduleConfig,
            loss_config = BASE_LOSS_CONFIG,
            start_scheduling = 50,
            device="cpu",
            using_multi_gpu = False,
            checkpoint_path = "./training/checkpoint",
            checkpoint_rate_unit = "epoch",
            checkpoint_rate:int = 1,
            early_stoping_strategy = "training_loss",
            patient = 5
            ):
        super().__init__(
            model, 
            optimizer(model.parameters(),**optimizer_config), 
            train_dataloader, 
            torch.device(device) if isinstance(device,str) else device
        )
        self.val_dataloader = validate_dataloader

        if main_scheduler is not None:
            self.main_scheduler = main_scheduler(self.optimizer,**main_scheduler_config)
        else:
            self.main_scheduler = None
        if warm_up_scheduler is not None:
            self.warm_up_scheduler = warm_up_scheduler(self.optimizer,**warm_up_scheduler_config)
        else:
            self.warm_up_scheduler = None
        
        self.start_scheduling = start_scheduling
        self.using_multi_gpu = using_multi_gpu
        if self.using_multi_gpu and check_device_is_cuda(self.device):
            self.device_ids = get_all_cuda_device_ids()
        else: 
            self.using_multi_gpu = False
            
        self.checkpoint_path = checkpoint_path if checkpoint_path[-1] != '/' else checkpoint_path[:-1]

        self.PIT_SISNR_mag_loss = PIT_SISNR_mag(self.device,**loss_config['PIT_SISNR_mag'])
        self.PIT_SISNR_time_loss = PIT_SISNR_time(self.device,**loss_config['PIT_SISNR_time'])
        self.PIT_SISNRi_loss = PIT_SISNRi(self.device,**loss_config['PIT_SISNRi'])
        self.PIT_SDRi_loss = PIT_SDRi(self.device,**loss_config['PIT_SDRi'])

        self.checkpoint_rate_unit = checkpoint_rate_unit
        assert self.checkpoint_rate_unit in ['epoch','iteration'],"only accept epoch or iteration checkpoint rate unit"
        self.checkpoint_rate = checkpoint_rate
        assert self.checkpoint_rate >= 1, "expect checkpoint_rate >= 1"

        self.iteration = 0
        self.early_stoping = early_stoping_strategy
        self.patient = patient

    def epoch_iteration(self,epoch,start_time,time_limit):
        self.model.train()
        tot_loss_time, num_batch = 0, 0
        pbar = tqdm(total=len(self.dataloader), unit='batches', bar_format='{l_bar}{bar:25}{r_bar}{bar:-10b}', colour="YELLOW", dynamic_ncols=True)
        tot_loss_freq = [0 for _ in range(self.model.num_stages)]
        for data in self.dataloader:
            mixture = data['mix']
            src = data['src']
            num_batch+=1
            input_sizes = mixture.size(-1)
            self.iteration+=1
            pbar.update(1)
            if epoch == 1: self.warmup_scheduler.step()
            model_input = mixture.to(self.device)
            self.optimizer.zero_grad()
            if self.using_multi_gpu:
                estim_src, estim_src_bn = torch.nn.parallel.data_parallel(self.model, model_input,device_ids=self.device_ids)
            else:
                estim_src, estim_src_bn = self.model(model_input)
            cur_loss_s_bn = 0
            cur_loss_s_bn = []
            for idx, estim_src_value in enumerate(estim_src_bn):
                cur_loss_s_bn.append(self.PIT_SISNR_mag_loss(estims=estim_src_value, idx=idx, input_sizes=input_sizes, target_attr=src))
                tot_loss_freq[idx] += cur_loss_s_bn[idx].item() / (self.model.num_spks)
            cur_loss_s = self.PIT_SISNR_time_loss(estims=estim_src, input_sizes=input_sizes, target_attr=src)
            if cur_loss_s.isnan():
                print('---------------------nan------------------------')
                tot_loss_time+= tot_loss_time/(num_batch-1)
                del cur_loss_s_bn, mixture, src, cur_loss_s, estim_src, estim_src_bn
                torch.cuda.empty_cache()
                gc.collect()
                bug = {
                    "model": self.model,
                    "PIT_SISNR_mag_loss":self.PIT_SISNR_mag_loss,
                    "tot_loss_freq": tot_loss_freq,
                    "cur_loss_s_bn": cur_loss_s_bn,
                    "PIT_SISNR_time_loss": self.PIT_SISNR_time_loss,
                    "mixture": mixture,
                    "src": src,
                    "output": (estim_src, estim_src_bn)
                }
                return bug
            tot_loss_time += cur_loss_s.item() / self.model.num_spks
            alpha = 0.4 * 0.8**(1+(epoch-101)//5) if epoch > 100 else 0.4
            cur_loss = (1-alpha) * cur_loss_s + alpha * sum(cur_loss_s_bn) / len(cur_loss_s_bn)
            cur_loss = cur_loss / self.model.num_spks
            cur_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),5.0)
            self.optimizer.step()
            dict_loss = {"T_Loss": tot_loss_time / num_batch}
            dict_loss.update({'F_Loss_' + str(idx): loss / num_batch for idx, loss in enumerate(tot_loss_freq)})
            pbar.set_postfix(dict_loss)
            del cur_loss_s_bn, mixture, src, cur_loss_s, alpha, cur_loss, estim_src, estim_src_bn, dict_loss
            torch.cuda.empty_cache()
            gc.collect()
            if time.time() - start_time > time_limit:
                print('-------------------out of time-----------------------')
                break
        pbar.close()
        tot_loss_freq = sum(tot_loss_freq) / len(tot_loss_freq)
        return tot_loss_time / num_batch, tot_loss_freq / num_batch, num_batch
    
    def validate(self):
        with torch.no_grad():
            self.model.eval()
            tot_loss_time, num_batch = 0, 0
            pbar = tqdm(total=len(self.val_dataloader), unit='batches', bar_format='{l_bar}{bar:25}{r_bar}{bar:-10b}', colour="YELLOW", dynamic_ncols=True)
            tot_loss_freq = [0 for _ in range(self.model.num_stages)]
            for data in self.val_dataloader:
                mixture = data['mix']
                src = data['src']
                num_batch+=1
                input_sizes = mixture.size(-1)
                self.iteration+=1
                pbar.update(1)
                model_input = mixture.to(self.device)
                if self.using_multi_gpu:
                    estim_src, estim_src_bn = torch.nn.parallel.data_parallel(self.model, model_input,device_ids=self.device_ids)
                else:
                    estim_src, estim_src_bn = self.model(model_input)
                cur_loss_s_bn = 0
                cur_loss_s_bn = []
                for idx, estim_src_value in enumerate(estim_src_bn):
                    cur_loss_s_bn.append(self.PIT_SISNR_mag_loss(estims=estim_src_value, idx=idx, input_sizes=input_sizes, target_attr=src))
                    tot_loss_freq[idx] += cur_loss_s_bn[idx].item() / (self.model.num_spks)
                cur_loss_s = self.PIT_SISNR_time_loss(estims=estim_src, input_sizes=input_sizes, target_attr=src)
                if cur_loss_s.isnan():
                    print('---------------------nan------------------------')
                    tot_loss_time+= tot_loss_time/(num_batch-1)
                    del cur_loss_s_bn, mixture, src, cur_loss_s, estim_src, estim_src_bn
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                tot_loss_time += cur_loss_s.item() / self.model.num_spks
                dict_loss = {"T_Loss":tot_loss_time / num_batch}
                dict_loss.update({'F_Loss_' + str(idx): loss / num_batch for idx, loss in enumerate(tot_loss_freq)})
                pbar.set_postfix(dict_loss)
                del cur_loss_s_bn, mixture, src, cur_loss_s, estim_src, estim_src_bn, dict_loss
                torch.cuda.empty_cache()
                gc.collect()
            pbar.close()
            tot_loss_freq = sum(tot_loss_freq) / len(tot_loss_freq)
            return tot_loss_time / num_batch, tot_loss_freq / num_batch, num_batch

    def train(self, epochs, time_limit, **kwargs):
        self.model.train()
        init_loss_time, init_loss_freq = 0, 0
        best_loss = 1000.0
        start_time = time.time()
        count = 0
        self.model.to(self.device)
        for epoch in range(1+epochs):
            valid_loss_best = init_loss_time
            train_start_time = time.time()
            try:
                bug = self.epoch_iteration(epoch,start_time,time_limit)
                train_loss_src_time, train_loss_src_freq, train_num_batch = bug
            except:
                return bug
            train_end_time = time.time()
            valid_start_time = time.time()
            valid_loss_src_time, valid_loss_src_freq, valid_num_batch = self.validate()
            valid_end_time = time.time()

            if epoch > self.start_scheduling:
                self.main_scheduler.step(valid_loss_src_time)
            print(f"[TRAIN] Loss(time/mini-batch) \n - Epoch {epoch:2d}: Loss_t = {train_loss_src_time:.4f} dB | Loss_f = {train_loss_src_freq:.4f} dB | Speed = ({train_end_time - train_start_time:.2f}s/{train_num_batch:d})")
            print(f"[VALID] Loss(time/mini-batch) \n - Epoch {epoch:2d}: Loss_t = {valid_loss_src_time:.4f} dB | Loss_f = {valid_loss_src_freq:.4f} dB | Speed = ({valid_end_time - valid_start_time:.2f}s/{valid_num_batch:d})")

            if self.checkpoint_rate_unit == 'epoch' and epoch%self.checkpoint_rate == 0:
                if valid_loss_src_time < best_loss:
                    best_loss = valid_loss_src_time

                    torch.save(self.model.state_dict(),f"{self.checkpoint_path}/model.pth")
                    torch.save(self.optimizer.state_dict(),f"{self.checkpoint_path}/opt.pth")
                    count = 0
                else:
                    count +=1
            
            if count > self.patient:
                print('early stoping because loss is not decay')
                break

            if time.time() > time_limit:
                print("-------------out of time------------------")
                break 
        return self.model
        
