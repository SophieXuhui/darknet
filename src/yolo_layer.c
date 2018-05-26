#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes)
{
    int i;
    layer l = {0};
    l.type = YOLO;

    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(total*2, sizeof(float));
    if(mask) l.mask = mask;
    else{
        l.mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + 4 + 1);
    l.inputs = l.outputs;
    l.truths = 90*(4 + 1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_yolo_layer;
    l.backward = backward_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_yolo_layer_gpu;
    l.backward_gpu = backward_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "yolo\n");
    srand(0);

    return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

// 获取预测出的box偏移，转为bounding box且归一化
// b.x、b.y: 
//      b.x、b.y，已归一化的bounding box的中心坐标。
//      x[]，相对cell左上角的偏移，已经做过LOGISTIC回归的归一化的值，对应paper公式中的tx、ty，即预测出的坐标偏移量；
//      i、j，当前cell相对feature map左上角的偏移，是cell的左上角坐标；对应paper公式中的cx、cy,注意是cell的左上角坐标非中心;
//      i + x[], 即预测出的box中心，相对feature map左上角的偏移;
//      lw、lh，当前feature map宽、高；除以lw、lh进行归一化。
// b.w、b.h:
//      b.w、b.h，已归一化的bounding box的宽高。
//      x[]，预测出的box的宽高变化量，exp(x[])宽高变化的缩放因子，有求导计算缩放因子的含义。
//      biases, anchor的宽高；这里，yolov3的[yolo]层中，anchor的宽高是相对网络输入尺度的，因此归一化时除以网络输入宽高w、h
//      w、h，网络输入的宽高，除以w、h，进行归一化
box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw; // b.x为归一化的值
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

// 根据ground truth和预测值，计算box的dalta
float delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = box_iou(pred, truth);

    // 将ground truth转为相对proposal的偏移量（坐标偏移、宽高缩放尺度）
    // 这里的truth已经相对featuremap归一化了（即经voc_label.py转为labels了），则先换算到featuremap尺度，再计算偏移量
    float tx = (truth.x*lw - i);
    float ty = (truth.y*lh - j);
    float tw = log(truth.w*w / biases[2*n]);  // biases是相对网络输入的尺度，所以乘以了网络尺度w（*w）
    float th = log(truth.h*h / biases[2*n + 1]);

    // 分别计算4个delta
    // scale: 2 - truth.w*truth.h
    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}


void delta_yolo_class(float *output, float *delta, int index, int class, int classes, int stride, float *avg_cat)
{
    int n;
    if (delta[index]){
        delta[index + stride*class] = 1 - output[index + stride*class];
        if(avg_cat) *avg_cat += output[index + stride*class];
        return;
    }
    for(n = 0; n < classes; ++n){
        delta[index + stride*n] = ((n == class)?1 : 0) - output[index + stride*n];
        if(n == class && avg_cat) *avg_cat += output[index + stride*n];
    }
}

static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}

void forward_yolo_layer(const layer l, network net)
{
    int i,j,b,t,n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

    // LOGISTIC回归计算box的x,y,以及obj、class的置信度；如果GPU, 是用forward_yolo_layer_gpu()函数回归计算
    // LOGISTIC: f(x) = 1/(1 + e^(-x)), 输出值归一化到0～1，且可以0.5划分为2类。x = 0, f=0.5; x < 0, 0 < f < 0.5; x > 0,  0.5 < f < 1
    // logistic一般用于二分类，而softmax是多分类，参考https://blog.csdn.net/kevin_123c/article/details/51684971
    // softmax的使用参考forward_region_layer()
    
    // l.output中信息存储顺序：平面结构，依次存x、y、w、h、objness、class1—conf，class2-conf……即，先存所有box的x值，长度为stride=l.w*l.h
    // 假设l.w*l.h = 2*2, 3个anchor，则：
    // xxxxyyyywwwwhhhhoooocccc xxxxyyyywwwwhhhhoooocccc xxxxyyyywwwwhhhhoooocccc
#ifndef GPU  
    for (b = 0; b < l.batch; ++b){ // 一个batch的图片数，由cfg配置的batch/subdivision计算（32/8 = 4），parse.c -> parse_net_options()
        for(n = 0; n < l.n; ++n){  // 当前yolo层的anchor个数, 即cfg中yolo层的mask定义的anchor的个数
            int index = entry_index(l, b, n*l.w*l.h, 0);  // 0，box的rect的索引
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);  // 计算box的x,y（乘2）, 处理featuremap上所有box的x,y值（乘l.w*l.h）
            index = entry_index(l, b, n*l.w*l.h, 4);      // 4，box的objness的索引
            activate_array(l.output + index, (1+l.classes)*l.w*l.h, LOGISTIC);  // 计算box的objness、class置信度
        }
    }
#endif
    
    // 清空上一轮batch训练的delta值
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    
    // 训练时net.train=1，测试时net.train=0
    if(!net.train) return; 
    
    // 计算一个batch的cost
    // delta： 先计算预测结果与GT的差值delta，分别是box的：x,y,w,h,objectness,class_conf
    // cost：  综合各类的delta计算cost
    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    
    // 计算一个batch上的delta
    
    for (b = 0; b < l.batch; ++b) {         // 逐张处理一个batch的图片
        
        // step 1
        // 先将预测出的每个box，均默认为非目标(非目标数量大)，并计算objectness的delta
        for (j = 0; j < l.h; ++j) {         
            for (i = 0; i < l.w; ++i) {     // 当前featuremap的高, l.w为宽
                for (n = 0; n < l.n; ++n) { // 当前yolo层的anchor数
                    
                    // step 1.1 : 对当前box与所有ground truth比较寻找best_iou
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);  // 计算box在预测结果中的开始位置索引
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.w*l.h); // 获取预测的box，归一化的x,y,w,h
                    
                    float best_iou = 0;
                    int best_t = 0;
                    for(t = 0; t < l.max_boxes; ++t){  // max_boxes :一张图片中最多的GT的box数，默认：90，parse.c  -> parse_yolo()
                        box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);  // net.truth存了一个batch中所有GT box的坐标和类别（4+1）
                        if(!truth.x) break; // l.truths:一张图中最多的GT box的信息buffer（坐标和类别），90*(4+1) , make_yolo_ layer(), region分支是30*（l.coords+1）
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {  
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    
                    // step 1.2  : 计算objectness的delta
                    // l.output[obj_index]：是否目标置信度值，已logistic规范化到0～1，0.5为中心
                    // 默认检测到的都是非目标，delta = 0 - l.output[obj_index]
                    // 如果objectness越小，这里默认为非目标是更准的，则delta越小，（但若后面step2通过iou将其判断为obj，会修改delta = 1-objectness），变大
                    // 如果objectness越大，这里默认为非目标是不准的，则delta越大，（同上，delta = 1-objectness，变小）
                    // 若best_iou满足阈值（yolov3.cfg设为0.7，代码默认0.5），与GT重合度很高，
                    // 则直接认为该pred box预测很准，是目标obj，则delta为0, 无需考虑objectness值
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    avg_anyobj += l.output[obj_index];
                    l.delta[obj_index] = 0 - l.output[obj_index];  // 0, 默认检测到的都是非目标
                    if (best_iou > l.ignore_thresh) {
                        l.delta[obj_index] = 0;
                    }
                    
                    // l.truth_thresh 输入为1，代码不执行
                    if (best_iou > l.truth_thresh) {
                        l.delta[obj_index] = 1 - l.output[obj_index];

                        int class = net.truth[best_t*(4 + 1) + b*l.truths + 4];
                        if (l.map) class = l.map[class];
                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                        delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, 0);
                        box truth = float_to_box(net.truth + best_t*(4 + 1) + b*l.truths, 1);
                        delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
                    }
                }
            }
        }
        
        // step 2
        // 对每个GT，与所有anchor（不仅是当前层anchor）中找best_iou
        // 通过坐标偏移来消除坐标的影响，通过计算iou,找宽高比和大小最接近的
        for(t = 0; t < l.max_boxes; ++t){
            box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w); // 当前GT中心坐标(truth.x * l.w)所在cell的左上角坐标，因为强制转为了int型，向下取整截断为左上角坐标
            j = (truth.y * l.h); 
            box truth_shift = truth;            // 输入的truth是相对featuremap归一化的（voc_label.py计算的labels)
            truth_shift.x = truth_shift.y = 0;  // 偏移到与anchor一样的位置，以cell的左上角点为中心点坐标
            for(n = 0; n < l.total; ++n){
                box pred = {0};
                pred.w = l.biases[2*n]/net.w;
                pred.h = l.biases[2*n+1]/net.h;
                float iou = box_iou(pred, truth_shift); // 计算anchor与GT的best_iou
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }

            // 在当前yolo层的mask中获取best_n对应的anchor的索引，有可能为-1,因为best_n不一定是在当前yolo检测层
            // best_n, 是相对所有anchor的索引
            // mask_n，是相对当前yolo层的anchor的索引，anchor[mask_n] --> best_n
            int mask_n = int_index(l.mask, best_n, l.n); 
            
            // 当前GT，best_n在当前层的mask中
            if(mask_n >= 0){
                int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);   // 与该anchor对应的、最接近该GT的box，i、j是GT的中心坐标
                
                // 计算gt与pred的iou、box的delta
                float iou = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
                
                // 计算objness的delta
                // 通过iou判断，认为当前pred的box是obj
                // 如果objness值越大，说明当前预测的越准，则delta越小
                // 如果objness值越小，说明当前预测的不准，则delta越大
                int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                avg_obj += l.output[obj_index];
                l.delta[obj_index] = 1 - l.output[obj_index];

                // 计算类别的delta
                // 各个类别都计算delta，预测的class_conf已logistic归一化到0～1
                // 已通过前面iou计算，认为当前pred的box的类别是GT的类别
                // 与GT一样的类别，delta = 1 - class_conf，class_conf越大（接近1）,delta越小
                // 与GT不同的类别, delta = 0 - class_conf,
                int class = net.truth[t*(4 + 1) + b*l.truths + 4];  // GT的类别
                if (l.map) class = l.map[class];
                int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1); // 预测的class置信度值位置索引
                delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, &avg_cat);

                ++count;
                ++class_count;
                if(iou > .5) recall += 1;   // pred与GT的iou
                if(iou > .75) recall75 += 1;
                avg_iou += iou;
            }
        }
    }
    
    //  计算一个batch的cost
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", net.index, avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count, count);
}

void backward_yolo_layer(const layer l, network net)
{
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 4 + 1; ++z){
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0){
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for(i = 0; i < l.outputs; ++i){
        l.output[i] = (l.output[i] + flip[i])/2.;
    }
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,j,n;
    float *predictions = l.output;
    if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for(j = 0; j < l.classes; ++j){
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_gpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_yolo_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_yolo_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

