class Config(object):
    backbone = 'resnet18'    # 'resnet18', 'resnet50', 'resnet101', 'resnet152'
    num_classes = 123
    metric = 'add_margin'    # 'add_margin', 'arc_margin', 'sphere'
    easy_margin = False
    use_se = True
    loss = 'CrossEntropyLoss'  # 'focal_loss','CrossEntropyLoss'

    train_list = '/home/hzy/Documents/work/pytorch_kaoqing_res18_fusion/img_list/train_rgb.list'
    val_list = '/home/hzy/Documents/work/pytorch_kaoqing_res18_fusion/img_list/test_rgb.list'

    checkpoints_path = 'checkpoints'
    save_interval = 5
    test_model_path = "/home/gp/work/project/pytorch_kaoqing/checkpoints/resnet18_20.pth"

    train_batch_size = 16  # batch size
    test_batch_size = 16

    input_shape = (3, 112, 112)

    optimizer = 'sgd'  # 'sgd', 'adam'

    use_gpu = True  # use GPU or not
    gpu_id = '0'
    num_workers = 4  # how many workers for loading data
    print_freq = 40  # print info every N batch
    test_freq = 500  # test model every N batch

    max_epoch = 20
    lr = 1e-1  # initial learning rate
    lr_step = 3
    #lr_decay = 0.9  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-5
