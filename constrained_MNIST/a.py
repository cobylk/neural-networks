from activations import EpochAwareMixin

a = EpochAwareMixin(total_epochs=100, decay_epochs=10, decay_schedule='exponential', decay_rate=0.4)

for i in range(15):
    a.current_epoch = i
    print(a.get_decay_factor(100., 1.0))
