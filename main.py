import time
import datetime

from unroller_gan import UnrollerGan
from kl_div import *

if torch.cuda.is_available():
    cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
else:
    cuda = False
torch.set_num_threads(2)

tick1 = time.time()

gan = UnrollerGan(cuda)

#load_gd(gan.G, gan.D, "test4")
gan.train(25000, 1000)
#kl_div_comp(cuda, gan.dset, gan.G, gan.D, gan.g_inp)

tick2 = time.time()
print("total time: " + str(datetime.timedelta(seconds=(tick2 - tick1))))
