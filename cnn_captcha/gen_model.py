import tensorflow as tf
import numpy as np
from gen_image import text_to_array, gen_require_captcha_image
from config import MAX_CAPTCHA, CHAR_SET_LEN, IMAGE_HEIGHT, IMAGE_WIDTH, MAX_ACCURACY

#申请占位符号
x_input = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
y_input = tf.placeholder(tf.float32, [None, CHAR_SET_LEN * MAX_CAPTCHA])
#keep_prob参数控制dropout几率
keep_prob = tf.placeholder(tf.float32)


# 把彩色图像转为灰度图像
def convert2gray(img):
    #图片维度等于3为彩色rpg图像,2为灰度图像
    if len(img.shape) > 2:
        #降维处理图片,转换为灰度图像
        gray = np.mean(img, -1)
        return gray
    else:
        return img

#定义两个函数进行初始化
#初始化权重1
def __weight_variable(shape, stddev=0.01):
    initial = tf.random_normal(shape, stddev=stddev)
    return tf.Variable(initial)

#初始化权重2
def __bias_variable(shape, stddev=0.1):
    initial = tf.random_normal(shape=shape, stddev=stddev)
    return tf.Variable(initial)

#定义卷积模板 1步长 ，0边距
def __conv2d(x, w):
    # strides 代表移动的平长
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

#定义池化模板,2x2的模板
def __max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 生成100张图片作为一个批次
def gen_next_batch(batch_size=100):
    #创建两个0数组
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])
    #将训练批次转为灰度图像
    for i in range(batch_size):
        text, image = gen_require_captcha_image()

        # 转成灰度图片
        image = convert2gray(image)
        #将图片信息转为向量,添加到0数组
        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text_to_array(text)

    return batch_x, batch_y


def create_layer(x_input, keep_prob):
    #将占位符x转为图4D向量,第2，第3维对应图片的宽，高，最后一维颜色通道数
    #灰度图像通道数为1
    x_image = tf.reshape(x_input, shape=[-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])

    # 定义第1个卷积层,在5x5的图片区域,在该区域提取32个特征
    w_c1 = __weight_variable([5, 5, 1, 32], stddev=0.1)
    b_c1 = __bias_variable([32], stddev=0.1)
    h_c1 = tf.nn.relu(tf.nn.bias_add(__conv2d(x_image, w_c1), b_c1))
    #第一层池化层
    h_pool1 = __max_pool_2x2(h_c1)


    # 定义第2个卷积层,在5x5的图片区域,在该区域提取64个特征
    w_c2 = __weight_variable([5, 5, 32, 64], stddev=0.1)
    b_c2 = __bias_variable([64], stddev=0.1)
    h_c2 = tf.nn.relu(tf.nn.bias_add(__conv2d(h_pool1, w_c2), b_c2))
    #第二层池化
    h_pool2 = __max_pool_2x2(h_c2)


    # 定义第3个卷积层,在5x5的图片区域,在该区域提取64个特征
    w_c3 = __weight_variable([5, 5, 64, 64], stddev=0.1)
    b_c3 = __bias_variable([64], stddev=0.1)
    h_c3 = tf.nn.relu(tf.nn.bias_add(__conv2d(h_pool2, w_c3), b_c3))
    #第三层池化
    h_pool3 = __max_pool_2x2(h_c3)



    #3层池化后图片减小到20x8
    # 密集连接层 ,把池化层输出的张量塑层向量,乘上权重矩阵
    w_fc1 = __weight_variable([20 * 8 * 64, 1024], stddev=0.1)
    #加入1024个神经元
    b_fc1 = __bias_variable([1024])
    #把池化层输出的张量塑层向量
    h_pool3_flat = tf.reshape(h_pool3, [-1, w_fc1.get_shape().as_list()[0]])
    #乘上权重矩阵
    h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool3_flat, w_fc1), b_fc1))
    #tf.nn.dropout()此函数是为了防止在训练中过拟合的操作，将训练输出按一定规则进行变换,将模型数字随机置0,keep_prob=1,忽略该操作
    h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

    # 输出层
    w_output = __weight_variable([1024, MAX_CAPTCHA * CHAR_SET_LEN], stddev=0.1)
    b_output = __bias_variable([MAX_CAPTCHA * CHAR_SET_LEN])
    y_output = tf.add(tf.matmul(h_fc1_dropout, w_output), b_output)

    return y_output

# 计算loss的典型方法
#loss 是估计值和真实值之映射到某一空间的误差
def create_loss(layer, y_input):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_input, logits=layer))
    return loss

# 计算正确率
def create_accuracy(output, y_input):
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(y_input, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def train():
    # 运行cnn
    layer_output = create_layer(x_input, keep_prob)
    #创建loss
    loss = create_loss(layer_output, y_input)
    #Tensorboar统计loss
    tf.summary.scalar('loss', loss)
    #创建正确率accuracy
    accuracy = create_accuracy(layer_output, y_input)
    #TensorBoard统计正确率
    tf.summary.scalar('accuracy', accuracy)
    global_step_tensor=tf.Variable(0,trainable="Flase",name="global_step")
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss,global_step=global_step_tensor)
    #保存模型
    saver = tf.train.Saver()

    with tf.Session() as sess:
        #tf.summary_merge_all()管理所有TensorBoard的summary
        merged = tf.summary.merge_all()
        #tf,summary.FileWriter()将TensorBoard数据保存
        writer = tf.summary.FileWriter('cnn_logs', sess.graph)
        tf.global_variables_initializer().run()
        #初始化正确率为0
        acc = 0.0
        i = 0

        while acc < MAX_ACCURACY:
            i += 1
            batch_x, batch_y = gen_next_batch(64)
            #计算loss
            _, _loss = sess.run([train_step, loss],
                                feed_dict={x_input: batch_x, y_input: batch_y, keep_prob: 0.5})

            # 每10次输出loss
            if i % 10 == 0:
                print(tf.train.global_step(sess,global_step_tensor), _loss)

            batch_x_test, batch_y_test = gen_next_batch(100)
            # 每100 step计算一次准确率并保存模型
            if i % 100 == 0:
                #计算正确率
                acc = sess.run(accuracy, feed_dict={x_input: batch_x_test, y_input: batch_y_test, keep_prob: 1.0})
                print('步长:%s' % i, '准确度: is %s' % acc)
                # 保存模型
                saver.save(sess, "./model/break.ckpt",global_step=i)
                # 如果准确率大于MAX_ACCURACY,完成训练
                if acc > MAX_ACCURACY:
                    print('正确率 > %s  ,停止计算' % MAX_ACCURACY)
                    break
            # 启用监控 tensor board
            summary = sess.run(merged, feed_dict={x_input: batch_x_test, y_input: batch_y_test, keep_prob: 1.})
            writer.add_summary(summary, i)

#单元测试模块
if __name__ == '__main__':
    train()
