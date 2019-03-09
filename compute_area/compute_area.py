import argparse
import tensorflow as tf
import csv
from RecordReaderAll import *
from utils import *
from scipy.io import loadmat, savemat
import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")

HEIGHT=192
WIDTH=256
NUM_PLANES = 20
NUM_THREADS = 4
numOutputPlanes = 20

def compute_semantic_id(semantics_i, new_element):
    new_semantic = semantics_i[new_element]
    elements_list = np.unique(new_semantic)
    max_numel = 0
    find_ele = 41
    for ele in elements_list:
        proposal_max_numel = np.sum(new_semantic == ele)
        if proposal_max_numel > max_numel:
            max_numel = proposal_max_numel
            find_ele = ele
    return find_ele, max_numel

def compute_location(new_xy):
    w = 256
    h = 192
    x = new_xy[0]
    y = new_xy[1]
    if x < h / 2:
        x_id = 0
    else:
        x_id = 1
    if y < w / 3:
        y_id = 1
    elif y < 2 * w / 3:
        y_id = 2
    else:
        y_id = 3

    location = x_id * 3 + y_id

    return location

def main(options):

    d = options.__dict__
    for key, value in d.iteritems():
        print '%s = %s' % (key, value)

    train_input = []
    train_input.append(options.dataFolder+'/planes_scannet_val.tfrecords')
    filename_queue_train = tf.train.string_input_producer(train_input, num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue_train)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            # 'height': tf.FixedLenFeature([], tf.int64),
            # 'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'image_path': tf.FixedLenFeature([], tf.string),
            'num_planes': tf.FixedLenFeature([], tf.int64),
            'plane': tf.FixedLenFeature([NUM_PLANES * 3], tf.float32),
            # 'plane_relation': tf.FixedLenFeature([NUM_PLANES * NUM_PLANES], tf.float32),
            'segmentation_raw': tf.FixedLenFeature([], tf.string),
            'depth': tf.FixedLenFeature([HEIGHT * WIDTH], tf.float32),
            'normal': tf.FixedLenFeature([HEIGHT * WIDTH * 3], tf.float32),
            'semantics_raw': tf.FixedLenFeature([], tf.string),
            'boundary_raw': tf.FixedLenFeature([], tf.string),
            'info': tf.FixedLenFeature([4 * 4 + 4], tf.float32),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image = tf.reshape(image, [HEIGHT, WIDTH, 3])

    depth = features['depth']
    depth = tf.reshape(depth, [HEIGHT, WIDTH, 1])

    normal = features['normal']
    normal = tf.cast(normal, tf.float32)
    normal = tf.reshape(normal, [HEIGHT, WIDTH, 3])

    normal = tf.nn.l2_normalize(normal, dim=2)

    # normal = tf.stack([normal[:, :, 1], normal[:, :, 0], normal[:, :, 2]], axis=2)

    semantics = tf.decode_raw(features['semantics_raw'], tf.uint8)
    semantics = tf.cast(tf.reshape(semantics, [HEIGHT, WIDTH]), tf.int32)

    numPlanes = tf.minimum(tf.cast(features['num_planes'], tf.int32), numOutputPlanes)

    numPlanesOri = numPlanes
    numPlanes = tf.maximum(numPlanes, 1)

    planes = features['plane']
    planes = tf.reshape(planes, [NUM_PLANES, 3])
    planes = tf.slice(planes, [0, 0], [numPlanes, 3])

    # shuffle_inds = tf.one_hot(tf.random_shuffle(tf.range(numPlanes)), depth = numPlanes)
    shuffle_inds = tf.one_hot(tf.range(numPlanes), numPlanes)

    planes = tf.transpose(tf.matmul(tf.transpose(planes), shuffle_inds))
    planes = tf.reshape(planes, [numPlanes, 3])
    planes = tf.concat([planes, tf.zeros([numOutputPlanes - numPlanes, 3])], axis=0)
    planes = tf.reshape(planes, [numOutputPlanes, 3])

    boundary = tf.decode_raw(features['boundary_raw'], tf.uint8)
    boundary = tf.cast(tf.reshape(boundary, (HEIGHT, WIDTH, 2)), tf.float32)

    # boundary = tf.decode_raw(features['boundary_raw'], tf.float64)
    # boundary = tf.cast(tf.reshape(boundary, (HEIGHT, WIDTH, 3)), tf.float32)
    # boundary = tf.slice(boundary, [0, 0, 0], [HEIGHT, WIDTH, 2])

    segmentation = tf.decode_raw(features['segmentation_raw'], tf.uint8)
    segmentation = tf.reshape(segmentation, [HEIGHT, WIDTH, 1])

    coef = tf.range(numPlanes)
    coef = tf.reshape(tf.matmul(tf.reshape(coef, [-1, numPlanes]), tf.cast(shuffle_inds, tf.int32)), [1, 1, numPlanes])

    plane_masks = tf.cast(tf.equal(segmentation, tf.cast(coef, tf.uint8)), tf.float32)
    plane_masks = tf.concat([plane_masks, tf.zeros([HEIGHT, WIDTH, numOutputPlanes - numPlanes])], axis=2)
    plane_masks = tf.reshape(plane_masks, [HEIGHT, WIDTH, numOutputPlanes])

    # non_plane_mask = tf.cast(tf.equal(segmentation, tf.cast(numOutputPlanes, tf.uint8)), tf.float32)
    non_plane_mask = 1 - tf.reduce_max(plane_masks, axis=2, keep_dims=True)
    # tf.cast(tf.equal(segmentation, tf.cast(numOutputPlanes, tf.uint8)), tf.float32)

    info = features['info']

    headers = ['image_id', 20, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    rows = []

    with tf.Session() as sess:

        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        i = 0
        try:
            while i < 1000:
                image_i, segmentation_i, semantics_i, numPlanes_i =\
                     sess.run([image, segmentation, semantics, numPlanes])

                segmentation_i = np.squeeze(segmentation_i)

                # cv2.imwrite(options.ImageSaveDir + '/' + str(i).zfill(7) + '_segmentation.png',
                #             drawSegmentationImage(segmentation_i, blackIndex=numOutputPlanes))
                # cv2.imwrite(options.ImageSaveDir + '/' + str(i).zfill(7) + '_semantics.png',
                #             drawSegmentationImage(semantics_i, blackIndex=numOutputPlanes))

                row = (str(i).zfill(7),)

                # area compute
                # new_element = np.sum(segmentation_i == 20)
                # row = row + (new_element,)
                # for j in range(numPlanes_i):
                #     new_element = np.sum(segmentation_i == j)
                #     row = row + (new_element,)

                # semantic_id compute
                # new_element = np.where(segmentation_i == 20)
                # find_ele, max_numel = compute_semantic_id(semantics_i, new_element)
                # row = row + (find_ele,)
                # for j in range(numPlanes_i):
                #     new_element = np.where(segmentation_i == j)
                #     find_ele, max_numel = compute_semantic_id(semantics_i, new_element)
                #     row = row + (find_ele,)

                # location compute
                new_element = np.where(segmentation_i == 20)
                new_xy = np.mean(new_element, axis=1)
                location = compute_location(new_xy)
                row = row + (location,)
                for j in range(numPlanes_i):
                    new_element = np.where(segmentation_i == j)
                    new_xy = np.mean(new_element, axis=1)
                    location = compute_location(new_xy)
                    row = row + (location,)

                rows.append(row)

                i = i+1

        except tf.errors.OutOfRangeError:
            print 'Done training -- epoch limit reached'

        finally:
            coord.request_stop()

        coord.join(threads)

    with open(options.CSVSaveDir3, 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PlaneNet param')

    parser.add_argument('--restore', type=int, default=0)
    parser.add_argument('--numOutputPlanes', type=int, default=20)
    parser.add_argument('--task', type=str, default='train')
    parser.add_argument('--ImageSaveDir', type=str, default='/home/xwj/github/data_image/sample2/seg_sem', help='path to the deep lab model')
    parser.add_argument('--CSVSaveDir', type=str, default='/home/xwj/github/data_image/sample2/csv/stocks.csv', help='path to csv')
    parser.add_argument('--CSVSaveDir2', type=str, default='/home/xwj/github/data_image/sample2/csv/semantic_id.csv', help='path to csv2')
    parser.add_argument('--CSVSaveDir3', type=str, default='/home/xwj/github/data_image/sample2/csv/location_id.csv', help='path to csv3')
    parser.add_argument('--dataFolder', type=str, default='/media/xwj/ScanNet/scannet_tfrecord', help='folder which contains tfrecords files')

    args = parser.parse_args()

    try:
        if args.task == "train":
            main(args)
        else:
            assert False,"format wrong"
            pass
    finally:
        pass