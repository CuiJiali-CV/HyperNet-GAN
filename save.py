import shutil


class saveModule(object):
    def __init__(self, category='Mnist', z_batch=16, img_batch=16, z_size=100, lr=0.001):
        self.z_batch = z_batch
        self.img_batch = img_batch
        self.category = category
        self.z_size = z_size
        self.lr = lr

        self.src_dir = './output/'
        self.des_dir = './output/history/z_batch{}_img_batch{}_2000{}/z{}_lr{}'.format(
            self.z_batch,
            self.img_batch,
            self.category,
            str(self.z_size),
            str(self.lr))

    def process(self):
        lst = [
            'checkpoint',
            'gens',
            'logs',
            'test'
        ]
        for dir in lst:
            src = self.src_dir + dir
            des = self.des_dir + '/' + dir
            shutil.copytree(src, des)
            shutil.rmtree(src)
