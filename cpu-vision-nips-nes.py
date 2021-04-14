from PIL import Image
from concurrent.futures import ThreadPoolExecutor
# from inception_v3_imagenet import model, SIZE
import numpy as np
from tensorflow.python.client import device_lib
from utils import *
import json
import pdb
import io
import os
import sys
import shutil
import time
import scipy.misc
import PIL
import csv
import random
import base64

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

# Imports the Google Cloud client library
import google.cloud.vision_v1 as vision
from google.cloud.vision_v1 import types

_GC_ACCESS_CONF = "hcaptcha-alpha-9c26e03ba9f2.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _GC_ACCESS_CONF

pool = ThreadPoolExecutor(max_workers=8)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from imagenet_labels import label_to_name
from local_dict import valid_label, valid_adv_label

# Instantiates a client
client = vision.ImageAnnotatorClient()

# Things you should definitely set:
IMAGENET_PATH = 'data/nips'
SOURCE_ID = sys.argv[2]
LABEL_INDEX = 6 # This is the colummn number of TrueLabel in the dev_dataset.csv for the NIPS data
OUT_DIR = "vision_nips_adv/"
EVALS_DIR = "vision_inputs/"
MOMENTUM = 0.5
# Things you can play around with:
# BATCH_SIZE = 20
# SIGMA = 1e-3
# EPSILON = 0.1
# EPS_DECAY = 0.005
# MIN_EPS_DECAY = 5e-5
# LEARNING_RATE = 1e-3
# SAMPLES_PER_DRAW = 500
# K = 15
# IMG_ID = sys.argv[1]
# MAX_LR = 1e-2
# MIN_LR = 5e-5
# Things you probably don't want to change:
# MAX_QUERIES = 4000000
# num_indices = 50000
# num_labels = 1000

# Things you can play around with:
BATCH_SIZE = 30
SIGMA = 1e-3
EPSILON = 0.05
EPS_DECAY = 0.005
MIN_EPS_DECAY = 5e-5
LEARNING_RATE = 1e-4
SAMPLES_PER_DRAW = 150
K = 15
IMG_ID = sys.argv[1]
MAX_LR = 1e-1
MIN_LR = 5e-4
# Things you probably don't want to change:
MAX_QUERIES = 4000000
num_indices = 50000
# num_labels = 1000

def main():
    evals_dir = EVALS_DIR
    out_dir = OUT_DIR
    k = K
    print('Starting partial-information attack with only top-' + str(k))
    target_image_id = SOURCE_ID
    source_class = label_to_name(int(data_lookup(target_image_id, LABEL_INDEX))).split(", ")[0]
    print("Target Image ID:", target_image_id)
    x, y = get_nips_dev_image(IMG_ID)
    orig_class = label_to_name(y).split(", ")[0]
    initial_img = x

    print("Setting img path to feed to gCloud Vision API")
    last_adv_img_path = os.path.join(os.path.join(IMAGENET_PATH, 'dev'), str(IMG_ID) + ".png")
    target_img = None
    target_img, _ = get_nips_dev_image(target_image_id)

    target_class = orig_class
    print('Set target class to be original img class %s for partial-info attack' % target_class)
    
    sess = tf.InteractiveSession()

    empty_dir(out_dir)
    empty_dir(evals_dir)

    batch_size = min(BATCH_SIZE, SAMPLES_PER_DRAW)
    assert SAMPLES_PER_DRAW % BATCH_SIZE == 0
    # one_hot_vec = one_hot(target_class, num_labels)

    x = tf.placeholder(tf.float32, initial_img.shape)
    x_t = tf.expand_dims(x, axis=0)
    good_inds = tf.placeholder(tf.int32)
    candidate_losses = tf.placeholder(tf.float32, [BATCH_SIZE])
    cpus = [get_available_cpus()[0]]
    print(cpus)
    # labels = np.repeat(np.expand_dims(one_hot_vec, axis=0), repeats=batch_size, axis=0)


    grad_estimates = []
    final_losses = []
    for i, device in enumerate(cpus):
        with tf.device(device):
            print('loading on cpu %d of %d' % (i+1, len(cpus)))
            noise_pos = tf.random_normal((batch_size//2,) + initial_img.shape)
            noise = tf.concat([noise_pos, -noise_pos], axis=0)
            eval_points = x_t + SIGMA * noise
            # logits, preds = model(sess, eval_points)
            # losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
        # vals, inds = tf.nn.top_k(logits, k=K)
        # inds is batch_size x k
        # good_inds = tf.where([valid_label(label, target_class) for label in candidate_labels]) # returns (# true) x 3
        # good_images = good_inds[:,0] # inds of img in batch that worked
        losses = tf.gather(candidate_losses, good_inds)
        noise = tf.gather(noise, good_inds)
        losses_tiled = tf.tile(tf.reshape(losses, (-1, 1, 1, 1)), (1,) + initial_img.shape)
        grad_estimates.append(tf.reduce_mean(losses_tiled * noise, \
            axis=0)/SIGMA)
        final_losses.append(losses)
    grad_estimate = tf.reduce_mean(grad_estimates, axis=0)
    final_losses = tf.concat(final_losses, axis=0)

    # eval network
    # with tf.device(cpus[0]):
    #     last_adv = os.path.join(evals_dir, '0.png')
    #     last_adv_labels = get_vision_labels(last_adv)
    #     # eval_logits, eval_preds = model(sess, x_t)
    #     eval_adv = tf.reduce_sum(tf.to_float([valid_label(label, target_class) for label in last_adv_labels])))

    samples_per_draw = SAMPLES_PER_DRAW
    def get_grad(pt, should_calc_truth=False):
        num_batches = samples_per_draw // batch_size
        losses = []
        grads = []
        feed_dict = {x: pt}
        for _ in range(num_batches):
            # candidate_path = os.path.join(evals_dir, '0.png')
            # scipy.misc.imsave(candidate_path, pt)
            candidates = sess.run(eval_points, feed_dict)
            futures = []
            c_labels = []
            c_losses = []
            start = time.time()
            gcv_successes = 0
            while (gcv_successes < BATCH_SIZE):
                for i in range(gcv_successes, BATCH_SIZE):
                    futures.append(pool.submit(get_vision_labels, candidates[i]))
                for future in futures:
                    response = future.result()
                    c_labels.append(response)
                    c_losses.append(combine_losses(source_class, response))
                gcv_successes = len(c_losses)
                if gcv_successes != BATCH_SIZE:
                    print("Successful responses:", gcv_successes)
            # for i in range(BATCH_SIZE):
            #     candidate = candidates[i]
            #     # candidate_path = os.path.join(evals_dir, '%s.png' % (i+1))
            #     # scipy.misc.imsave(candidate_path, candidate)
            #     c_label_group = get_vision_labels(candidate)
            #     # print("label " + str(i) + ": " + str(c_label_group))
            #     c_labels.append(c_label_group)
            #     c_losses.append(combine_losses(target_class, c_label_group))
            finish = time.time()
            print("Got image labels for batch ", _, ": ", str(finish - start))
            g_inds = np.where([sum([valid_adv_label(l.description, source_class) for l in label[:5]]) >= 1 for label in c_labels])[0]
            # print([[l.description for l in label] for label in c_labels])
            # print(np.array(c_labels).shape)
            # print("GINDS:", g_inds)
            # start = time.time()
            loss, dl_dx_ = sess.run([final_losses, grad_estimate], feed_dict={candidate_losses: c_losses, good_inds: g_inds})
            losses.append(np.mean(loss))
            grads.append(dl_dx_)
            # finish = time.time()
            # print("Calculated gradients: ", str(finish - start))
        return np.array(losses).mean(), np.mean(np.array(grads), axis=0)

    # with tf.device('/cpu:0'):
    #     render_feed = tf.placeholder(tf.float32, initial_img.shape)
    #     render_exp = tf.expand_dims(render_feed, axis=0)
    #     render_logits, _ = model(sess, render_exp)

    def render_frame(image, save_index):
        # actually draw the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
        # image
        ax1.imshow(image)
        fig.sca(ax1)
        plt.xticks([])
        plt.yticks([])
        # classifications
        # probs = softmax(sess.run(render_logits, {render_feed: image})[0])
        adv_labels = get_vision_labels(image)
        total_probs = sum([label.score for label in adv_labels])
        probs = np.array([label.score/total_probs for label in adv_labels])
        k_val = min(4, len(probs))
        topk = probs.argsort()[-k_val:][::-1]
        topprobs = probs[topk]
        barlist = ax2.bar(range(k_val), topprobs)
        topk_labels = []
        for i in range(len(topk)):
            index = topk[i]
            adv_label = adv_labels[index]
            topk_labels.append(adv_label)
            # if valid_label(adv_label.description, orig_class):
            #     barlist[i].set_color('g')
            if valid_adv_label(adv_label.description, source_class):
                barlist[i].set_color('r')
        # for i, v in enumerate(topk):
        #     if v == orig_class:
        #         barlist[i].set_color('g')
        #     if v == target_class:
        #         barlist[i].set_color('r')
        # print("TOPK:", [(l.description, l.score) for l in topk_labels])
        plt.sca(ax2)
        plt.ylim([0, 1.1])
        plt.xticks(range(k_val), [topk_labels[i].description[:15] for i in range(len(topk))], rotation='vertical')
        
        for bar, label in zip(barlist, [topk_labels[i].score for i in range(len(topk))]):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, label, ha='center', va='bottom')

        fig.subplots_adjust(bottom=0.2)

        path = os.path.join(out_dir, 'frame%06d.png' % save_index)
        if os.path.exists(path):
            os.remove(path)
        plt.savefig(path)
        plt.close()

    adv = initial_img.copy()
    assert out_dir[-1] == '/'

    log_file = open(os.path.join(out_dir, 'log.txt'), 'w+')
    g = 0
    num_queries = 0

    last_ls = []
    current_lr = LEARNING_RATE

    max_iters = int(np.ceil(MAX_QUERIES // SAMPLES_PER_DRAW))
    real_eps = 0.5
    
    lrs = []
    max_lr = MAX_LR
    epsilon_decay = EPS_DECAY
    last_good_adv = adv
    for i in range(max_iters):
        start = time.time()
        render_frame(adv, i)

        # see if we should stop
        # padv = sess.run(eval_adv, feed_dict={x: adv})
        # last_adv = os.path.join(evals_dir, '0.png')
        # if not os.path.exists(last_adv):
        #     last_adv = last_adv_img_path
        last_adv_labels = get_vision_labels(adv)
        if len(last_adv_labels) != 0:
            # eval_logits, eval_preds = model(sess, x_t)
            padv = valid_adv_label(last_adv_labels[0].description, source_class)
            if (padv == 1) and (real_eps <= EPSILON):
                 print('partial info early stopping at iter %d' % i)
                 break
        else:
            print("ANNEALING EPS DECAY")
            adv = last_good_adv # start over with a smaller eps
            l, g = get_grad(adv)
            assert (l < 1)
            epsilon_decay = max(epsilon_decay / 2, MIN_EPS_DECAY)

        assert target_img is not None
        lower = np.clip(target_img - real_eps, 0., 1.)
        upper = np.clip(target_img + real_eps, 0., 1.)
        prev_g = g
        l, g = get_grad(adv)

        if l < .1:
            real_eps = max(EPSILON, real_eps - epsilon_decay)
            max_lr = MAX_LR
            last_good_adv = adv
            epsilon_decay = EPS_DECAY
            if real_eps <= EPSILON:
                samples_per_draw = 5000
            last_ls = []

        # simple momentum
        g = MOMENTUM * prev_g + (1.0 - MOMENTUM) * g

        last_ls.append(l)
        last_ls = last_ls[-5:]
        if last_ls[-1] > last_ls[0] and len(last_ls) == 5:
            if max_lr > MIN_LR:
                print("ANNEALING MAX LR")
                max_lr = max(max_lr / 2.0, MIN_LR)
            else:
                print("ANNEALING EPS DECAY")
                adv = last_good_adv # start over with a smaller eps
                l, g = get_grad(adv)
                assert (l < 1)
                epsilon_decay = max(epsilon_decay / 2, MIN_EPS_DECAY)
            last_ls = []

            print("last_adv_labels:", last_adv_labels)
        # backtracking line search for optimal lr
        current_lr = max_lr
        while current_lr > MIN_LR:
            proposed_adv = adv - current_lr * np.sign(g)
            proposed_adv = np.clip(proposed_adv, lower, upper)
            num_queries += 1
            # eval_logits_ = sess.run(eval_logits, {x: proposed_adv})[0]
            target_class_in_top_k = sum([valid_adv_label(label.description, source_class) for label in last_adv_labels[:5]]) >= 1
            if target_class_in_top_k:
                lrs.append(current_lr)
                adv = proposed_adv
                break
            else:
                current_lr = current_lr / 2
                print('backtracking, lr = %.2E' % current_lr)

        num_queries += SAMPLES_PER_DRAW

        log_text = 'Step %05d: loss %.4f eps %.4f eps-decay %.4E lr %.2E (time %.4f)' % (i, l, \
                real_eps, epsilon_decay, current_lr, time.time() - start)
        log_file.write(log_text + '\n')
        print(log_text)

        np.save(os.path.join(out_dir, '%s.npy' % (i+1)), adv)
        last_adv_img_path = os.path.join(out_dir, '%s.png' % (i+1))
        img = PIL.Image.fromarray((adv*255).astype('uint8'), mode="RGB")        
        img.save(last_adv_img_path)
    pool.shutdown()

def pseudorandom_target_id():
    data_path = os.path.join(IMAGENET_PATH, 'dev')
    file = random.choice(os.listdir(data_path))
    filename, file_extension = os.path.splitext(file)
    return filename

def data_lookup(id, index):
    labels_path = os.path.join(IMAGENET_PATH, 'dev_dataset.csv')
    with open(labels_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == id:
                return row[index]

def get_nips_dev_image(id):
    data_path = os.path.join(IMAGENET_PATH, 'dev')
    labels_path = os.path.join(IMAGENET_PATH, 'dev_dataset.csv')
    def get(id):
        path = os.path.join(data_path, str(id) + ".png")
        x = load_image(path)
        print('labels_path:', labels_path)
        y = data_lookup(id, LABEL_INDEX) 
        return x, int(y)
    return get(id)

# get center crop
def load_image(path):
    image = PIL.Image.open(path)
    if image.height > image.width:
        height_off = int((image.height - image.width)/2)
        image = image.crop((0, height_off, image.width, height_off+image.width))
    elif image.width > image.height:
        width_off = int((image.width - image.height)/2)
        image = image.crop((width_off, 0, width_off+image.height, image.height))
    image = image.resize((299, 299))
    img = np.asarray(image).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.repeat(img[:,:,np.newaxis], repeats=3, axis=2)
    if img.shape[2] == 4:
        # alpha channel
        img = img[:,:,:3]
    return img

def get_vision_labels(img, print_labels=False):
    # Loads the image into memory
    pil_img = Image.fromarray(np.uint8(img*255))
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    # with io.open(target_image_path, 'rb') as image_file:
    #     content = image_file.read()

    vision_target_image = types.Image(content=img_byte_arr)
    vision_response = client.label_detection(image=vision_target_image)
    labels = vision_response.label_annotations
        
    if print_labels:
        print('Labels (and confidence score):')
        print('=' * 30)
        for label in vision_response.label_annotations:
            print(label.description, '(%.2f%%)' % (label.score*100.))
    return labels

def combine_losses(source_class, labels):
    total_score = sum([label.score for label in labels])
    return sum([(label.score * (1-valid_adv_label(label.description, source_class)))/total_score for label in labels])

def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']

def empty_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

if __name__ == '__main__':
    main()
