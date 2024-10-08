from torch.optim.lr_scheduler import LambdaLR


class Scheduler(object):
    """Container for warmup and normal scheduler."""
    def __init__(self, normal_schededuler, warmup_scheduler=None):
        self.warmup = warmup_scheduler
        self.sched = normal_schededuler

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler."""
        if self.warmup is not None and not self.warmup.complete:
            return self.warmup.get_last_lr()

        return self.sched.get_last_lr()

    def state_dict(self):
        """Returns the state of each scheduler."""
        state_dict = {
            'warmup': self.warmup.state_dict() if self.warmup is not None else {},
            'sched': self.sched.state_dict(),

        }
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state"""
        if self.warmup:
            self.warmup.load_state_dict(state_dict['warmup'])

        self.sched.load_state_dict(state_dict['sched'])

    def step(self, *args, **kwargs):
        if self.warmup is not None and not self.warmup.complete:
            return self.warmup.step(*args, **kwargs)

        return self.sched.step(*args, **kwargs)


class LinearWarmup(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over warmup steps 
        and keep learning rate 1 after the warmup steps
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.complete = False
        super(LinearWarmup, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step == 0:
            return (float(step+1) / float(max(1.0, self.warmup_steps)))*0.5
            
        if step < self.warmup_steps:
            return (float(step) / float(max(1.0, self.warmup_steps)))

        self.complete = True
        return 1.0
