#include "darknet.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

extern void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top);
extern void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen);
extern void run_yolo(int argc, char **argv);
extern void run_detector(int argc, char **argv);
extern void run_coco(int argc, char **argv);
extern void run_captcha(int argc, char **argv);
extern void run_nightmare(int argc, char **argv);
extern void run_classifier(int argc, char **argv);
extern void run_regressor(int argc, char **argv);
extern void run_segmenter(int argc, char **argv);
extern void run_char_rnn(int argc, char **argv);
extern void run_tag(int argc, char **argv);
extern void run_cifar(int argc, char **argv);
extern void run_go(int argc, char **argv);
extern void run_art(int argc, char **argv);
extern void run_super(int argc, char **argv);
extern void run_lsd(int argc, char **argv);

void average(int argc, char *argv[])
{
    char *cfgfile = argv[2];
    char *outfile = argv[3];
    gpu_index = -1;
    network *net = parse_network_cfg(cfgfile);
    network *sum = parse_network_cfg(cfgfile);

    char *weightfile = argv[4];   
    load_weights(sum, weightfile);

    int i, j;
    int n = argc - 5;
    for(i = 0; i < n; ++i){
        weightfile = argv[i+5];   
        load_weights(net, weightfile);
        for(j = 0; j < net->n; ++j){
            layer l = net->layers[j];
            layer out = sum->layers[j];
            if(l.type == CONVOLUTIONAL){
                int num = l.n*l.c*l.size*l.size;
                axpy_cpu(l.n, 1, l.biases, 1, out.biases, 1);
                axpy_cpu(num, 1, l.weights, 1, out.weights, 1);
                if(l.batch_normalize){
                    axpy_cpu(l.n, 1, l.scales, 1, out.scales, 1);
                    axpy_cpu(l.n, 1, l.rolling_mean, 1, out.rolling_mean, 1);
                    axpy_cpu(l.n, 1, l.rolling_variance, 1, out.rolling_variance, 1);
                }
            }
            if(l.type == CONNECTED){
                axpy_cpu(l.outputs, 1, l.biases, 1, out.biases, 1);
                axpy_cpu(l.outputs*l.inputs, 1, l.weights, 1, out.weights, 1);
            }
        }
    }
    n = n+1;
    for(j = 0; j < net->n; ++j){
        layer l = sum->layers[j];
        if(l.type == CONVOLUTIONAL){
            int num = l.n*l.c*l.size*l.size;
            scal_cpu(l.n, 1./n, l.biases, 1);
            scal_cpu(num, 1./n, l.weights, 1);
                if(l.batch_normalize){
                    scal_cpu(l.n, 1./n, l.scales, 1);
                    scal_cpu(l.n, 1./n, l.rolling_mean, 1);
                    scal_cpu(l.n, 1./n, l.rolling_variance, 1);
                }
        }
        if(l.type == CONNECTED){
            scal_cpu(l.outputs, 1./n, l.biases, 1);
            scal_cpu(l.outputs*l.inputs, 1./n, l.weights, 1);
        }
    }
    save_weights(sum, outfile);
}

long numops(network *net)
{
    int i;
    long ops = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            ops += 2l * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w;
        } else if(l.type == CONNECTED){
            ops += 2l * l.inputs * l.outputs;
        } else if (l.type == RNN){
            ops += 2l * l.input_layer->inputs * l.input_layer->outputs;
            ops += 2l * l.self_layer->inputs * l.self_layer->outputs;
            ops += 2l * l.output_layer->inputs * l.output_layer->outputs;
        } else if (l.type == GRU){
            ops += 2l * l.uz->inputs * l.uz->outputs;
            ops += 2l * l.uh->inputs * l.uh->outputs;
            ops += 2l * l.ur->inputs * l.ur->outputs;
            ops += 2l * l.wz->inputs * l.wz->outputs;
            ops += 2l * l.wh->inputs * l.wh->outputs;
            ops += 2l * l.wr->inputs * l.wr->outputs;
        } else if (l.type == LSTM){
            ops += 2l * l.uf->inputs * l.uf->outputs;
            ops += 2l * l.ui->inputs * l.ui->outputs;
            ops += 2l * l.ug->inputs * l.ug->outputs;
            ops += 2l * l.uo->inputs * l.uo->outputs;
            ops += 2l * l.wf->inputs * l.wf->outputs;
            ops += 2l * l.wi->inputs * l.wi->outputs;
            ops += 2l * l.wg->inputs * l.wg->outputs;
            ops += 2l * l.wo->inputs * l.wo->outputs;
        }
    }
    return ops;
}

void speed(char *cfgfile, int tics)
{
    if (tics == 0) tics = 1000;
    network *net = parse_network_cfg(cfgfile);
    set_batch_network(net, 1);
    int i;
    double time=what_time_is_it_now();
    image im = make_image(net->w, net->h, net->c*net->batch);
    for(i = 0; i < tics; ++i){
        network_predict(net, im.data);
    }
    double t = what_time_is_it_now() - time;
    long ops = numops(net);
    printf("\n%d evals, %f Seconds\n", tics, t);
    printf("Floating Point Operations: %.2f Bn\n", (float)ops/1000000000.);
    printf("FLOPS: %.2f Bn\n", (float)ops/1000000000.*tics/t);
    printf("Speed: %f sec/eval\n", t/tics);
    printf("Speed: %f Hz\n", tics/t);
}

void operations(char *cfgfile)
{
    gpu_index = -1;
    network *net = parse_network_cfg(cfgfile);
    long ops = numops(net);
    printf("Floating Point Operations: %ld\n", ops);
    printf("Floating Point Operations: %.2f Bn\n", (float)ops/1000000000.);
}

void oneoff(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = parse_network_cfg(cfgfile);
    int oldn = net->layers[net->n - 2].n;
    int c = net->layers[net->n - 2].c;
    scal_cpu(oldn*c, .1, net->layers[net->n - 2].weights, 1);
    scal_cpu(oldn, 0, net->layers[net->n - 2].biases, 1);
    net->layers[net->n - 2].n = 11921;
    net->layers[net->n - 2].biases += 5;
    net->layers[net->n - 2].weights += 5*c;
    if(weightfile){
        load_weights(net, weightfile);
    }
    net->layers[net->n - 2].biases -= 5;
    net->layers[net->n - 2].weights -= 5*c;
    net->layers[net->n - 2].n = oldn;
    printf("%d\n", oldn);
    layer l = net->layers[net->n - 2];
    copy_cpu(l.n/3, l.biases, 1, l.biases +   l.n/3, 1);
    copy_cpu(l.n/3, l.biases, 1, l.biases + 2*l.n/3, 1);
    copy_cpu(l.n/3*l.c, l.weights, 1, l.weights +   l.n/3*l.c, 1);
    copy_cpu(l.n/3*l.c, l.weights, 1, l.weights + 2*l.n/3*l.c, 1);
    *net->seen = 0;
    save_weights(net, outfile);
}

void oneoff2(char *cfgfile, char *weightfile, char *outfile, int l)
{
    gpu_index = -1;
    network *net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights_upto(net, weightfile, 0, net->n);
        load_weights_upto(net, weightfile, l, net->n);
    }
    *net->seen = 0;
    save_weights_upto(net, outfile, net->n);
}

void partial(char *cfgfile, char *weightfile, char *outfile, int max)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 1);
    save_weights_upto(net, outfile, max);
}

void print_weights(char *cfgfile, char *weightfile, int n)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 1);
    layer l = net->layers[n];
    int i, j;
    //printf("[");
    for(i = 0; i < l.n; ++i){
        //printf("[");
        for(j = 0; j < l.size*l.size*l.c; ++j){
            //if(j > 0) printf(",");
            printf("%g ", l.weights[i*l.size*l.size*l.c + j]);
        }
        printf("\n");
        //printf("]%s\n", (i == l.n-1)?"":",");
    }
    //printf("]");
}

void rescale_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            rescale_weights(l, 2, -.5);
            break;
        }
    }
    save_weights(net, outfile);
}

void rgbgr_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            rgbgr_weights(l);
            break;
        }
    }
    save_weights(net, outfile);
}

void reset_normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == CONVOLUTIONAL && l.batch_normalize) {
            denormalize_convolutional_layer(l);
        }
        if (l.type == CONNECTED && l.batch_normalize) {
            denormalize_connected_layer(l);
        }
        if (l.type == GRU && l.batch_normalize) {
            denormalize_connected_layer(*l.input_z_layer);
            denormalize_connected_layer(*l.input_r_layer);
            denormalize_connected_layer(*l.input_h_layer);
            denormalize_connected_layer(*l.state_z_layer);
            denormalize_connected_layer(*l.state_r_layer);
            denormalize_connected_layer(*l.state_h_layer);
        }
    }
    save_weights(net, outfile);
}

layer normalize_layer(layer l, int n)
{
    int j;
    l.batch_normalize=1;
    l.scales = calloc(n, sizeof(float));
    for(j = 0; j < n; ++j){
        l.scales[j] = 1;
    }
    l.rolling_mean = calloc(n, sizeof(float));
    l.rolling_variance = calloc(n, sizeof(float));
    return l;
}

void normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL && !l.batch_normalize){
            net->layers[i] = normalize_layer(l, l.n);
        }
        if (l.type == CONNECTED && !l.batch_normalize) {
            net->layers[i] = normalize_layer(l, l.outputs);
        }
        if (l.type == GRU && l.batch_normalize) {
            *l.input_z_layer = normalize_layer(*l.input_z_layer, l.input_z_layer->outputs);
            *l.input_r_layer = normalize_layer(*l.input_r_layer, l.input_r_layer->outputs);
            *l.input_h_layer = normalize_layer(*l.input_h_layer, l.input_h_layer->outputs);
            *l.state_z_layer = normalize_layer(*l.state_z_layer, l.state_z_layer->outputs);
            *l.state_r_layer = normalize_layer(*l.state_r_layer, l.state_r_layer->outputs);
            *l.state_h_layer = normalize_layer(*l.state_h_layer, l.state_h_layer->outputs);
            net->layers[i].batch_normalize=1;
        }
    }
    save_weights(net, outfile);
}

void statistics_net(char *cfgfile, char *weightfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == CONNECTED && l.batch_normalize) {
            printf("Connected Layer %d\n", i);
            statistics_connected_layer(l);
        }
        if (l.type == GRU && l.batch_normalize) {
            printf("GRU Layer %d\n", i);
            printf("Input Z\n");
            statistics_connected_layer(*l.input_z_layer);
            printf("Input R\n");
            statistics_connected_layer(*l.input_r_layer);
            printf("Input H\n");
            statistics_connected_layer(*l.input_h_layer);
            printf("State Z\n");
            statistics_connected_layer(*l.state_z_layer);
            printf("State R\n");
            statistics_connected_layer(*l.state_r_layer);
            printf("State H\n");
            statistics_connected_layer(*l.state_h_layer);
        }
        printf("\n");
    }
}

void denormalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if ((l.type == DECONVOLUTIONAL || l.type == CONVOLUTIONAL) && l.batch_normalize) {
            denormalize_convolutional_layer(l);
            net->layers[i].batch_normalize=0;
        }
        if (l.type == CONNECTED && l.batch_normalize) {
            denormalize_connected_layer(l);
            net->layers[i].batch_normalize=0;
        }
        if (l.type == GRU && l.batch_normalize) {
            denormalize_connected_layer(*l.input_z_layer);
            denormalize_connected_layer(*l.input_r_layer);
            denormalize_connected_layer(*l.input_h_layer);
            denormalize_connected_layer(*l.state_z_layer);
            denormalize_connected_layer(*l.state_r_layer);
            denormalize_connected_layer(*l.state_h_layer);
            l.input_z_layer->batch_normalize = 0;
            l.input_r_layer->batch_normalize = 0;
            l.input_h_layer->batch_normalize = 0;
            l.state_z_layer->batch_normalize = 0;
            l.state_r_layer->batch_normalize = 0;
            l.state_h_layer->batch_normalize = 0;
            net->layers[i].batch_normalize=0;
        }
    }
    save_weights(net, outfile);
}

void mkimg(char *cfgfile, char *weightfile, int h, int w, int num, char *prefix)
{
    network *net = load_network(cfgfile, weightfile, 0);
    image *ims = get_weights(net->layers[0]);
    int n = net->layers[0].n;
    int z;
    for(z = 0; z < num; ++z){
        image im = make_image(h, w, 3);
        fill_image(im, .5);
        int i;
        for(i = 0; i < 100; ++i){
            image r = copy_image(ims[rand()%n]);
            rotate_image_cw(r, rand()%4);
            random_distort_image(r, 1, 1.5, 1.5);
            int dx = rand()%(w-r.w);
            int dy = rand()%(h-r.h);
            ghost_image(r, im, dx, dy);
            free_image(r);
        }
        char buff[256];
        sprintf(buff, "%s/gen_%d", prefix, z);
        save_image(im, buff);
        free_image(im);
    }
}

void visualize(char *cfgfile, char *weightfile)
{
    network *net = load_network(cfgfile, weightfile, 0);
    visualize_network(net);
#ifdef OPENCV
    cvWaitKey(0);
#endif
}

int main(int argc, char **argv)
{
    //test_resize("data/bad.jpg");
    //test_box();
    //test_convolutional_layer();
    if(argc < 2){
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }
    gpu_index = find_int_arg(argc, argv, "-i", 0);
    if(find_arg(argc, argv, "-nogpu")) {
        gpu_index = -1;
    }

#ifndef GPU
    gpu_index = -1;
#else
    if(gpu_index >= 0){
        cuda_set_device(gpu_index);
    }
#endif
    // 相比之前版本
    // 新增： print、segmenter
    // 减少：voxel、vid_rnn、compare、dice、writing
    if (0 == strcmp(argv[1], "average")){  // average: 一个网络，多个权重文件，计算平均权重文件，按层按元素逐个加和、再平均
        average(argc, argv);               // ./darknet average xx.cfg outfile weight1 weight2 weight3 …… 
    } else if (0 == strcmp(argv[1], "yolo")){ // yolo：单线程, 无随机缩放，REGION_DATA
        run_yolo(argc, argv);                 // REGION_DATA: load_data_region加载数据,数据增广由jitter控制随机crop，超出部分复制原边界，无缩放
    } else if (0 == strcmp(argv[1], "super")){
        run_super(argc, argv);
    } else if (0 == strcmp(argv[1], "lsd")){
        run_lsd(argc, argv);
    } else if (0 == strcmp(argv[1], "detector")){ // detector: 支持多GPU，支持随机缩放, DETECTION_DATA，与“yolo”对比
        run_detector(argc, argv);                 // DETECTION_DATA：load_data_detection加载数据,jitter和scale控制随机crop和缩放，且按长边缩放，不足补0.5
    } else if (0 == strcmp(argv[1], "detect")){   // detect ：几乎同detector命令，这里指定用cfg/coco.data
        float thresh = find_float_arg(argc, argv, "-thresh", .5);
        char *filename = (argc > 4) ? argv[4]: 0;
        char *outfile = find_char_arg(argc, argv, "-out", 0);
        int fullscreen = find_arg(argc, argv, "-fullscreen");
        test_detector("cfg/coco.data", argv[2], argv[3], filename, thresh, .5, outfile, fullscreen);
    } else if (0 == strcmp(argv[1], "cifar")){
        run_cifar(argc, argv);
    } else if (0 == strcmp(argv[1], "go")){
        run_go(argc, argv);
    } else if (0 == strcmp(argv[1], "rnn")){
        run_char_rnn(argc, argv);
    } else if (0 == strcmp(argv[1], "coco")){
        run_coco(argc, argv);
    } else if (0 == strcmp(argv[1], "classify")){   // classify : 分类预测调用实例，几乎同classifier，这里指定用cfg/imagenet1k.data进行分类预测
        predict_classifier("cfg/imagenet1k.data", argv[2], argv[3], argv[4], 5);
    } else if (0 == strcmp(argv[1], "classifier")){ // classifier: 分类的训练、预测调用接口，
        run_classifier(argc, argv);
    } else if (0 == strcmp(argv[1], "regressor")){
        run_regressor(argc, argv);
    } else if (0 == strcmp(argv[1], "segmenter")){
        run_segmenter(argc, argv);
    } else if (0 == strcmp(argv[1], "art")){
        run_art(argc, argv);
    } else if (0 == strcmp(argv[1], "tag")){
        run_tag(argc, argv);
    } else if (0 == strcmp(argv[1], "3d")){
        composite_3d(argv[2], argv[3], argv[4], (argc > 5) ? atof(argv[5]) : 0);
    } else if (0 == strcmp(argv[1], "test")){
        test_resize(argv[2]);
    } else if (0 == strcmp(argv[1], "captcha")){
        run_captcha(argc, argv);
    } else if (0 == strcmp(argv[1], "nightmare")){
        run_nightmare(argc, argv);
    } else if (0 == strcmp(argv[1], "rgbgr")){ // rgbgr: 换conv层权重的通道顺序，darknet加载、处理是R/G/B顺序，只换im.c==3的层，一般为图像输入层
        rgbgr_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "reset")){
        reset_normalize_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "denormalize")){
        denormalize_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "statistics")){
        statistics_net(argv[2], argv[3]);
    } else if (0 == strcmp(argv[1], "normalize")){
        normalize_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "rescale")){
        rescale_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "ops")){ // ops：计算网络卷积层浮点数计算量flops，只要输入cfg文件即可：./darknet ops xxx.cfg
        operations(argv[2]);
    } else if (0 == strcmp(argv[1], "speed")){ // speed：计算网络前向时间，单GPU, 默认前向1000次计算，只卷积，不包含region层等框回归部分
        speed(argv[2], (argc > 3 && argv[3]) ? atoi(argv[3]) : 0);  // ./darknet speed xxx*.cfg N（N可指定的计算次数）
    } else if (0 == strcmp(argv[1], "oneoff")){ // oneoff： 卷积输出层（region前一层），权重、bias ？？？
        oneoff(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "oneoff2")){
        oneoff2(argv[2], argv[3], argv[4], atoi(argv[5]));
    } else if (0 == strcmp(argv[1], "print")){  // print: 屏幕显示指定层的weights值，根据kenel数按行打印，每行kenel.w * kenel.h * input_chnl
        print_weights(argv[2], argv[3], atoi(argv[4]));  //./darknet print xx.cfg xx.weights N (层序号，从0开始，包括res、route等层）
    } else if (0 == strcmp(argv[1], "partial")){ // partial：从已有权重文件中获取部分，获取从第0层开始到给定第N层的权重，保存输出
        partial(argv[2], argv[3], argv[4], atoi(argv[5]));
    } else if (0 == strcmp(argv[1], "average")){ // average: 一个网络，多个权重文件，计算平均权重文件，按层按元素逐个加和、再平均
        average(argc, argv);                     // ./darknet average xx.cfg outfile weight1 weight2 weight3 ……
    } else if (0 == strcmp(argv[1], "visualize")){  // visualize: 图片形式显示每一卷积层的卷积核参数（个数、值分布），值归一化再*255，.png
        visualize(argv[2], (argc > 3) ? argv[3] : 0); // ./darknet visualize xx.cfg  xx.weights（可有可无）
    } else if (0 == strcmp(argv[1], "mkimg")){        // mkimg ： 根据网络和权重，生成第一层卷积的权重的图 ？？
        mkimg(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), argv[7]); // ./darknet visualize xx.cfg  xx.weights h w num path
    } else if (0 == strcmp(argv[1], "imtest")){  // imtest ：对输入图像多种变换的测试，计算图像像素幅值L2范数，图像灰度，hsv多种变化等
        test_resize(argv[2]);       // ./darknet imtest  xx.jpg
    } else {
        fprintf(stderr, "Not an option: %s\n", argv[1]);
    }
    return 0;
}

