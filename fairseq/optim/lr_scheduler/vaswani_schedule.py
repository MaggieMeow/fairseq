# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('vaswani')
class InverseSquareRootSchedule(FairseqLRScheduler):
    """Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`--warmup-init-lr`) until the configured
    learning rate (`--lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup:

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup:

      lr = decay_factor / sqrt(update_num)

    where

      decay_factor = args.lr * sqrt(args.warmup_updates)
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with inverse_sqrt.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        warmup_end_lr = args.lr[0]
        if args.warmup_init_lr < 0:
            args.warmup_init_lr = warmup_end_lr
        else:
            args.warmup_init_lr = 1.0 * args.warmup_updates**-1.5
            print('init lr before d_model is', args.warmup_init_lr)

        # linearly warmup for the first args.warmup_updates
        self.warmup_init_lr = args.warmup_updates**-1.5

        # initial learning rate
        self.d_model = args.hidden_layer_size
        self.d_model_factor = self.d_model**-0.5
        self.lr = args.warmup_init_lr * self.d_model_factor
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        parser.add_argument('--warmup-updates', default=4000, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--warmup-init-lr', default=0, type=float, metavar='LR',
                            help='default is 0.')
        parser.add_argument('--hidden-layer-size', default=1024, type=int, metavar='HL')

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
       # if num_updates < self.args.warmup_updates:
       #     self.lr = (self.args.warmup_init_lr + num_updates*self.lr_step) * self.d_model**-0.5
       # else:
       #     self.lr = self.d_model**-0.5 * num_updates**-0.5
        lr1 = num_updates * self.warmup_init_lr
        lr2 = num_updates ** -0.5

        self.lr = min([lr1, lr2]) * self.d_model_factor
        
        self.optimizer.set_lr(self.lr)
        return self.lr
