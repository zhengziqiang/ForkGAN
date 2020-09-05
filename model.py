from collections import namedtuple
from utils import *
from ops import *
import time
from glob import glob
def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer+noise

def generator_resnet(image, options, reuse=False, name="generator"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        def residule_block_dilated(x, dim, ks=3, s=1, name='res'):
            y = instance_norm(dilated_conv2d(x, dim, ks, s, padding='SAME', name=name + '_c1'), name + '_bn1')
            y = tf.nn.relu(y)
            y = instance_norm(dilated_conv2d(y, dim, ks, s, padding='SAME', name=name + '_c2'), name + '_bn2')
            return y + x

        ### Encoder architecture
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim * 2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim * 4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        r1 = residule_block_dilated(c3, options.gf_dim * 4, name='g_r1')
        r2 = residule_block_dilated(r1, options.gf_dim * 4, name='g_r2')
        r3 = residule_block_dilated(r2, options.gf_dim * 4, name='g_r3')
        r4 = residule_block_dilated(r3, options.gf_dim * 4, name='g_r4')
        r5 = residule_block_dilated(r4, options.gf_dim * 4, name='g_r5')

        ### translation decoder architecture
        r6 = residule_block_dilated(r5, options.gf_dim * 4, name='g_r6')
        r7 = residule_block_dilated(r6, options.gf_dim * 4, name='g_r7')
        r8 = residule_block_dilated(r7, options.gf_dim * 4, name='g_r8')
        r9 = residule_block_dilated(r8, options.gf_dim * 4, name='g_r9')
        d1 = deconv2d(r9, options.gf_dim * 2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))

        ### reconstruction decoder architecture
        r5 = gaussian_noise_layer(r5, 0.02)
        r6_rec = residule_block_dilated(r5, options.gf_dim * 4, name='g_r6_rec')
        r6_rec = gaussian_noise_layer(r6_rec, 0.02)
        r7_rec = residule_block_dilated(r6_rec, options.gf_dim * 4, name='g_r7_rec')
        r8_rec = residule_block_dilated(r7_rec, options.gf_dim * 4, name='g_r8_rec')
        r9_rec = residule_block_dilated(r8_rec, options.gf_dim * 4, name='g_r9_rec')
        d1_rec = deconv2d(r9_rec, options.gf_dim * 2, 3, 2, name='g_d1_dc_rec')
        d1_rec = tf.nn.relu(instance_norm(d1_rec, 'g_d1_bn_rec'))
        d2_rec = deconv2d(d1_rec, options.gf_dim, 3, 2, name='g_d2_dc_rec')
        d2_rec = tf.nn.relu(instance_norm(d2_rec, 'g_d2_bn_rec'))
        d2_rec = tf.pad(d2_rec, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred_rec = tf.nn.tanh(conv2d(d2_rec, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c_rec'))

        return pred,pred_rec,r5

def domain_agnostic_classifier(percep,options, reuse=False,name="percep"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        h1 = lrelu(instance_norm(conv2d(percep, options.df_dim * 4, name='d_h1_conv'), 'd_bn1'))
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim * 2, name='d_h2_conv'), 'd_bn2'))
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim * 2, name='d_h3_conv'), 'd_bn3'))
        h4 = conv2d(h3, 2, s=1, name='d_h3_pred')
        return tf.reshape(tf.reduce_mean(h4,axis=[0,1,2]),[-1,1,1,2])

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_ - target) ** 2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))



def mae_criterion_list(in_, target):
    loss = 0.0
    for i in range(len(target)):
        loss+=tf.reduce_mean((in_[i]-target[i])**2)
    return loss / len(target)


def sce_criterion_list(logits, labels):
    loss = 0.0
    for i in range(len(labels)):
        loss+=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits[i], labels=labels[i]))
    return loss/len(labels)

epsilon = 1e-9

class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir
        self.n_d = args.n_d
        self.n_scale = args.n_scale
        self.ndf= args.ndf
        self.load_size =args.load_size
        self.fine_size =args.fine_size
        self.generator = generator_resnet
        self.domain_agnostic_classifier=domain_agnostic_classifier
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
            self.criterionGAN_list = mae_criterion_list
        else:
            self.criterionGAN = sce_criterion
            self.criterionGAN_list = sce_criterion_list

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf//args.n_d, args.output_nc,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(args.max_size)

    def discriminator(self,image, options, reuse=False, name="discriminator"):
        images = []
        for i in range(self.n_scale):
            images.append(tf.image.resize_bicubic(image, [get_shape(image)[1]//(2**i),get_shape(image)[2]//(2**i)]))
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            images = dis_down(images,4,2,self.n_scale,options.df_dim,'d_h0_conv_scale_')
            images = dis_down(images,4,2, self.n_scale, options.df_dim*2, 'd_h1_conv_scale_')
            images = dis_down(images,4,2, self.n_scale, options.df_dim * 4, 'd_h2_conv_scale_')
            images = dis_down(images,4,2, self.n_scale, options.df_dim * 8, 'd_h3_conv_scale_')
            images = final_conv(images,self.n_scale,"d_pred_scale_")
            return images

    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size*2,self.input_c_dim + self.output_c_dim],name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        A_label = np.zeros([1, 1, 1, 2], dtype=np.float32)
        B_label = np.zeros([1, 1, 1, 2], dtype=np.float32)
        A_label[:, :, :, 0] = 1.0
        B_label[:, :, :, 1] = 1.0
        self.A_label=tf.convert_to_tensor(A_label)
        self.B_label=tf.convert_to_tensor(B_label)

        self.fake_B,self.rec_realA,self.realA_percep = self.generator(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A_,self.rec_fakeB,self.fakeB_percep = self.generator(self.fake_B, self.options, False, name="generatorB2A")
        self.fake_A,self.rec_realB,self.realB_percep = self.generator(self.real_B, self.options, True, name="generatorB2A")
        self.fake_B_,self.rec_fakeA,self.fakeA_percep = self.generator(self.fake_A, self.options, True, name="generatorA2B")

        self.realA_percep_logit =  self.domain_agnostic_classifier(self.realA_percep,self.options,  False,name="percep")
        self.realB_percep_logit =  self.domain_agnostic_classifier(self.realB_percep, self.options, True, name="percep")
        self.fakeA_percep_logit = self.domain_agnostic_classifier(self.fakeA_percep, self.options, True, name="percep")
        self.fakeB_percep_logit = self.domain_agnostic_classifier(self.fakeB_percep, self.options, True, name="percep")

        self.g_cls_loss=sce_criterion(self.fakeA_percep_logit,self.A_label)*0.5+sce_criterion(self.fakeB_percep_logit,self.B_label)*0.5
        self.cls_loss = sce_criterion(self.realA_percep_logit,self.A_label)*0.25+sce_criterion(self.realB_percep_logit,self.B_label)*0.25\
                        +sce_criterion(self.fakeB_percep_logit,self.A_label)*0.25+sce_criterion(self.fakeA_percep_logit,self.B_label)*0.25

        self.g_adv_total=0.0
        self.g_adv = 0.0
        self.g_adv_rec = 0.0
        self.g_adv_recfake = 0.0
        ### We switch to adopt the pixel-wise loss between the two latent codes after the reduce_mean operation
        self.percep_loss = tf.reduce_mean(tf.abs(tf.reduce_mean(self.realA_percep,axis=3)-tf.reduce_mean(self.fakeB_percep,axis=3)))\
                           +tf.reduce_mean(tf.abs(tf.reduce_mean(self.realB_percep,axis=3)-tf.reduce_mean(self.fakeA_percep,axis=3)))
        for i in range(self.n_d):
            self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name=str(i)+"_discriminatorB")
            self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name=str(i)+"_discriminatorA")

            self.DB_fake_rec = self.discriminator(self.rec_realB, self.options, reuse=False, name=str(i) + "_reconstructionB")
            self.DA_fake_rec = self.discriminator(self.rec_realA, self.options, reuse=False, name=str(i) + "_reconstructionA")

            self.fake_B_rec = self.discriminator(self.rec_fakeB, self.options, reuse=False, name=str(i) + "_refineB")
            self.fake_A_rec = self.discriminator(self.rec_fakeA, self.options, reuse=False, name=str(i) + "_refineA")

            self.g_adv_total+=((self.criterionGAN_list(self.DA_fake, get_ones_like(self.DA_fake))+ self.criterionGAN_list(self.DB_fake, get_ones_like(self.DB_fake)))*0.5+
                               (self.criterionGAN_list(self.fake_B_rec,get_ones_like(self.fake_B_rec))+self.criterionGAN_list(self.fake_A_rec,get_ones_like(self.fake_A_rec)))*0.5+
                               (self.criterionGAN_list(self.DA_fake_rec,get_ones_like(self.DA_fake_rec)) + self.criterionGAN_list(self.DB_fake_rec, get_ones_like(self.DB_fake_rec))) * 0.5)

            self.g_adv+=((self.criterionGAN_list(self.DA_fake, get_ones_like(self.DA_fake))+ self.criterionGAN_list(self.DB_fake, get_ones_like(self.DB_fake)))*0.5)
            self.g_adv_rec+=((self.criterionGAN_list(self.fake_B_rec,get_ones_like(self.fake_B_rec))+self.criterionGAN_list(self.fake_A_rec,get_ones_like(self.fake_A_rec)))*0.5)
            self.g_adv_recfake+=((self.criterionGAN_list(self.DA_fake_rec,get_ones_like(self.DA_fake_rec)) + self.criterionGAN_list(self.DB_fake_rec, get_ones_like(self.DB_fake_rec))) * 0.5)

        self.g_loss_a2b = self.criterionGAN_list(self.DB_fake, get_ones_like(self.DB_fake)) \
                          + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
                          + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss_b2a = self.criterionGAN_list(self.DA_fake, get_ones_like(self.DA_fake)) \
                          + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
                          + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss = self.g_adv_total+ self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
                      + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)\
                      + self.L1_lambda * abs_criterion(self.rec_realA, self.real_A)\
                      + self.L1_lambda * abs_criterion(self.rec_realB, self.real_B)+ self.percep_loss + self.g_cls_loss

        self.g_rec_real = abs_criterion(self.rec_realA, self.real_A) + abs_criterion(self.rec_realB, self.real_B)
        self.g_rec_cycle = abs_criterion(self.real_A, self.fake_A_) + abs_criterion(self.real_B, self.fake_B_)

        self.fake_A_sample = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size*2,  self.output_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size*2,  self.output_c_dim], name='fake_B_sample')
        self.rec_A_sample = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size * 2, self.output_c_dim],name='rec_A_sample')
        self.rec_B_sample = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size * 2, self.output_c_dim],name='rec_B_sample')
        self.rec_fakeA_sample = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size * 2,self.output_c_dim], name='rec_fakeA_sample')
        self.rec_fakeB_sample = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size * 2,self.output_c_dim], name='rec_fakeB_sample')

        self.d_loss_item=[]
        self.d_loss_item_rec = []
        self.d_loss_item_recfake = []

        for i in range(self.n_d):
            self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name=str(i)+"_discriminatorB")
            self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name=str(i)+"_discriminatorA")
            self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name=str(i)+"_discriminatorB")
            self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name=str(i)+"_discriminatorA")
            self.db_loss_real = self.criterionGAN_list(self.DB_real, get_ones_like(self.DB_real))
            self.db_loss_fake = self.criterionGAN_list(self.DB_fake_sample, get_zeros_like(self.DB_fake_sample))
            self.db_loss = (self.db_loss_real * 0.5 + self.db_loss_fake * 0.5)
            self.da_loss_real = self.criterionGAN_list(self.DA_real, get_ones_like(self.DA_real))
            self.da_loss_fake = self.criterionGAN_list(self.DA_fake_sample, get_zeros_like(self.DA_fake_sample))
            self.da_loss = (self.da_loss_real * 0.5 + self.da_loss_fake * 0.5)
            self.d_loss = (self.da_loss + self.db_loss)
            self.d_loss_item.append(self.d_loss)

            self.DB_real_rec = self.discriminator(self.real_B, self.options, reuse=True, name=str(i)+"_reconstructionB")
            self.DA_real_rec = self.discriminator(self.real_A, self.options, reuse=True, name=str(i)+"_reconstructionA")
            self.DB_fake_rec = self.discriminator(self.rec_B_sample, self.options, reuse=True,name=str(i) + "_reconstructionB")
            self.DA_fake_rec = self.discriminator(self.rec_A_sample, self.options, reuse=True,name=str(i) + "_reconstructionA")
            self.db_loss_real_rec = self.criterionGAN_list(self.DB_real_rec, get_ones_like(self.DB_real_rec))
            self.db_loss_fake_rec = self.criterionGAN_list(self.DB_fake_rec, get_zeros_like(self.DB_fake_rec))
            self.db_loss_rec = (self.db_loss_real_rec * 0.5 + self.db_loss_fake_rec * 0.5)
            self.da_loss_real_rec = self.criterionGAN_list(self.DA_real_rec, get_ones_like(self.DA_real_rec))
            self.da_loss_fake_rec = self.criterionGAN_list(self.DA_fake_rec, get_zeros_like(self.DA_fake_rec))
            self.da_loss_rec = (self.da_loss_real_rec * 0.5 + self.da_loss_fake_rec * 0.5)
            self.d_loss_rec = (self.da_loss_rec + self.db_loss_rec)
            self.d_loss_item_rec.append(self.d_loss_rec)


            self.real_B_adv = self.discriminator(self.real_B, self.options, reuse=True, name=str(i)+"_refineB")
            self.real_A_adv = self.discriminator(self.real_A, self.options, reuse=True, name=str(i)+"_refineA")
            self.DB_fake_recfake = self.discriminator(self.rec_fakeB_sample, self.options, reuse=True,name=str(i) + "_refineB")
            self.DA_fake_recfake = self.discriminator(self.rec_fakeA_sample, self.options, reuse=True,name=str(i) + "_refineA")
            self.db_loss_real_recfake = self.criterionGAN_list(self.real_B_adv, get_ones_like(self.real_B_adv))
            self.db_loss_fake_recfake = self.criterionGAN_list(self.DB_fake_recfake, get_zeros_like(self.DB_fake_recfake))
            self.da_loss_real_recfake = self.criterionGAN_list(self.real_A_adv, get_ones_like(self.real_A_adv))
            self.da_loss_fake_recfake = self.criterionGAN_list(self.DA_fake_recfake, get_zeros_like(self.DA_fake_recfake))
            self.db_loss_recfake = (self.db_loss_real_recfake * 0.5 + self.db_loss_fake_recfake * 0.5)
            self.da_loss_recfake = (self.da_loss_real_recfake * 0.5 + self.da_loss_fake_recfake * 0.5)
            self.d_loss_recfake = (self.db_loss_recfake + self.da_loss_recfake)
            self.d_loss_item_recfake.append(self.d_loss_recfake)

        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )

        self.test_A = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size*2,self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size*2,self.output_c_dim], name='test_B')

        self.testB,self.rec_testA,_ = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.rec_cycle_A,self.refine_testB,_ =self.generator(self.testB, self.options, True, name="generatorB2A")

        self.testA,self.rec_testB,_ = self.generator(self.test_B, self.options, True, name="generatorB2A")
        self.rec_cycle_B, self.refine_testA,_ = self.generator(self.testA, self.options, True, name="generatorA2B")

        t_vars = tf.trainable_variables()

        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.p_vars = [var for var in t_vars if 'percep' in var.name]
        self.d_vars_item=[]
        for i in range(self.n_d):
            self.d_vars=[var for var in t_vars if str(i)+'_discriminator' in var.name]
            self.d_vars_item.append(self.d_vars)

        self.R_vars_item = []
        for i in range(self.n_d):
            self.R_vars = [var for var in t_vars if str(i) + '_reconstruction' in var.name]
            self.R_vars_item.append(self.R_vars)

        self.refine_vars_item = []
        for i in range(self.n_d):
            self.refine_vars = [var for var in t_vars if str(i) + '_refine' in var.name]
            self.refine_vars_item.append(self.refine_vars)

        # for var in t_vars: print(var.name)

    def train(self, args):
        """Train cyclegan"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')

        ### generator
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        ### domain-agnostic classifier
        self.p_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.cls_loss, var_list=self.p_vars)

        ### translation
        self.d_optim_item=[]
        for i in range(self.n_d):
            self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
                .minimize(self.d_loss_item[i], var_list=self.d_vars_item[i])
            self.d_optim_item.append(self.d_optim)

        ### reconstruction
        self.rec_optim_item = []
        for i in range(self.n_d):
            self.rec_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
                .minimize(self.d_loss_item_rec[i], var_list=self.R_vars_item[i])
            self.rec_optim_item.append(self.rec_optim)

        ### refinement
        self.recfake_optim_item = []
        for i in range(self.n_d):
            self.recfake_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
                .minimize(self.d_loss_item_recfake[i], var_list=self.refine_vars_item[i])
            self.recfake_optim_item.append(self.recfake_optim)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(os.path.join(args.checkpoint_dir,"logs"), self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):
            dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size
            lr = args.lr if epoch < args.epoch_step else args.lr * (args.epoch - epoch) / (args.epoch - args.epoch_step)

            for idx in range(0, batch_idxs):
                batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))
                batch_images = [load_train_data(batch_file, args.load_size, args.fine_size) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)
                # Update G network and record fake outputs
                fake_A,fake_B,rec_A,rec_B,rec_fake_A,rec_fake_B,_,_,g_loss,gan_loss,g_rec_cycle,g_rec_real,percep,g_adv,g_adv_rec,g_adv_recfake,cls_loss,g_cls_loss,summary_str = self.sess.run(
                    [self.fake_A, self.fake_B,self.rec_realA,self.rec_realB,self.rec_fakeA,self.rec_fakeB, self.g_optim,self.p_optim,self.g_loss,self.g_adv_total,self.g_rec_cycle,self.g_rec_real,self.percep_loss,self.g_adv,self.g_adv_rec,self.g_adv_recfake,self.cls_loss,self.g_cls_loss,self.g_sum],
                    feed_dict={self.real_data: batch_images, self.lr: lr})
                self.writer.add_summary(summary_str, counter)
                [fake_A, fake_B] = self.pool([fake_A, fake_B])
                [rec_A, rec_B] = self.pool([rec_A, rec_B])
                [rec_fake_A, rec_fake_B] = self.pool([rec_fake_A, rec_fake_B])

                # Update D network
                loss_print=[]
                for i in range(self.n_d):
                    _, d_loss,d_sum = self.sess.run(
                        [self.d_optim_item[i], self.d_loss_item[i],self.d_sum],
                        feed_dict={self.real_data: batch_images,
                                   self.fake_A_sample: fake_A,
                                   self.fake_B_sample: fake_B,self.lr: lr})

                    _, d_loss_rec = self.sess.run(
                        [self.rec_optim_item[i], self.d_loss_item_rec[i]],
                        feed_dict={self.real_data: batch_images,
                                   self.rec_A_sample: rec_A,
                                   self.rec_B_sample: rec_B,self.lr: lr})

                    _, d_loss_recfake = self.sess.run(
                        [self.recfake_optim_item[i], self.d_loss_item_recfake[i]],
                        feed_dict={self.real_data: batch_images,
                                   self.rec_fakeA_sample: rec_fake_A,
                                   self.rec_fakeB_sample: rec_fake_B, self.lr: lr})

                    loss_print.append(d_loss)
                    loss_print.append(d_loss_rec)
                    loss_print.append(d_loss_recfake)

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f g_loss: %4.4f gan:%4.4f adv:%4.4f adv_rec:%4.4f adv_recfake:%4.4f pix_cycle:%4.4f pix_rec:%4.4f g_percep:%4.4f cls:%4.4f g_cls:%4.4f" % (
                    epoch, idx, batch_idxs, time.time() - start_time,g_loss,gan_loss,g_adv,g_adv_rec,g_adv_recfake,g_rec_cycle,g_rec_real,percep,cls_loss,g_cls_loss)))
                print(loss_print)

                if np.mod(counter, args.print_freq) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, args.save_freq) == 2:
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        np.random.shuffle(dataA)
        np.random.shuffle(dataB)
        batch_files = list(zip(dataA[:self.batch_size], dataB[:self.batch_size]))
        sample_images = [load_train_data(batch_file,self.load_size,self.fine_size, is_testing=True) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)

        fake_A, fake_B,rec_cycle_A,rec_cycle_B,rec_A,rec_B,rec_fakeA,rec_fakeB = self.sess.run(
            [self.fake_A, self.fake_B,self.fake_A_,self.fake_B_,self.rec_realA,self.rec_realB,self.rec_fakeA,self.rec_fakeB],
            feed_dict={self.real_data: sample_images}
        )
        real_A = sample_images[:, :, :, :3]
        real_B = sample_images[:, :, :, 3:]

        merge_A = np.concatenate([real_B, fake_A,rec_B,rec_fakeA,rec_cycle_B], axis=2)
        merge_B = np.concatenate([real_A, fake_B,rec_A,rec_fakeB,rec_cycle_A], axis=2)
        check_folder('./{}/{:02d}'.format(sample_dir, epoch))
        save_images(merge_A, [self.batch_size, 1],
                    './{}/{:02d}/A_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(merge_B, [self.batch_size, 1],
                    './{}/{:02d}/B_{:04d}.jpg'.format(sample_dir, epoch, idx))

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        out_var,refine_var, in_var,rec_var,cycle_var = (self.testB,self.refine_testB, self.test_A,self.rec_testA,self.rec_cycle_A) if args.which_direction == 'AtoB' else (
            self.testA,self.refine_testA, self.test_B,self.rec_testB,self.rec_cycle_B)
        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, args.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(args.test_dir,'{0}_{1}'.format(args.which_direction, os.path.basename(sample_file)))
            fake_img,refine_fake,rec_img,cycle_img = self.sess.run([out_var,refine_var,rec_var,cycle_var], feed_dict={in_var: sample_image})
            merge=np.concatenate([sample_image,fake_img,refine_fake,rec_img,cycle_img],axis=2)
            save_images(merge, [1, 1], image_path)