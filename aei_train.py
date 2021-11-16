import os
from omegaconf import OmegaConf
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint


from aei_net import Encoder

def main(args):
    hp = OmegaConf.load(args.config)
    model = Encoder(hp)
    save_path = os.path.join(hp.log.chkpt_dir, args.name)
    os.makedirs(save_path, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(hp.log.chkpt_dir, args.name),
        monitor='val_loss',
        verbose=True,
        save_last=True,
        save_top_k=args.save_top_k,  # save all
    )


    # trainer = Trainer(
    #     logger=pl_loggers.TensorBoardLogger(hp.log.log_dir),
    #     #early_stop_callback=None,
    #     checkpoint_callback=checkpoint_callback,
    #     weights_save_path=save_path,
    #     gpus=-1 if args.gpus is None else args.gpus,
    #     distributed_backend='ddp',
    #     num_sanity_val_steps=1,
    #     # resume_from_checkpoint=args.checkpoint_path,
    #     gradient_clip_val=hp.model.grad_clip,
    #     fast_dev_run=args.fast_dev_run,
    #     val_check_interval=args.val_interval,
    #     progress_bar_refresh_rate=5,
    #     max_epochs=10000,
    #     auto_scale_batch_size=True,
    #     accumulate_grad_batches=4,
    # )

    if hp.stage == 'train' or hp.stage == 'retrain':

        trainer = Trainer(
            logger=pl_loggers.TensorBoardLogger(hp.log.log_dir),
            #early_stop_callback=None,
            checkpoint_callback=checkpoint_callback,
            weights_save_path=save_path,
            gpus=-1 if args.gpus is None else args.gpus,
            distributed_backend='ddp',
            num_sanity_val_steps=1,
            # resume_from_checkpoint=args.checkpoint_path,
            gradient_clip_val=hp.model.grad_clip,
            fast_dev_run=args.fast_dev_run,
            val_check_interval=args.val_interval,
            progress_bar_refresh_rate=5,
            max_epochs=10000,
            auto_scale_batch_size=True,
            accumulate_grad_batches=4,
        )
    else:
        trainer = Trainer(
            gpus=-1 if args.gpus is None else args.gpus
        )
    if hp.stage == 'train':
        model = model.load_from_checkpoint(hp.log.start_checkpoint,hp=hp,strict=False)
        #model = model.load_from_checkpoint(args.checkpoint_path,hp=hp,strict=False)
        trainer.fit(model)
    elif hp.stage == 'retrain':
        trainer.fit(model)
        
    elif hp.stage == 'test':
        model = model.load_from_checkpoint(hp.log.test_checkpoint,hp=hp,strict=False)
        trainer.test(model)
        #trainer.test(model)
    


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="path of configuration yaml file")
    parser.add_argument('-g', '--gpus', type=str, default=None,
                        help="Number of gpus to use (e.g. '0,1,2,3'). Will use all if not given.")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="Name of the run.")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint for resuming")
    parser.add_argument('-s', '--save_top_k', type=int, default=1,
                        help="save top k checkpoints, default(-1): save all")
    parser.add_argument('-f', '--fast_dev_run', type=bool, default=False,
                        help="fast run for debugging purpose")
    parser.add_argument('--val_interval', type=float, default=0.002,
                        help="run val loop every * training epochs")

    args = parser.parse_args()

    main(args)
