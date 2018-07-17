变化：保存模型时，文件头

      函数：save_weights_upto()
      文件：parser.c
      改变：4个部分中最后一个，数据长度为 size_t ，在linux下，为64字节。而原始为int，32字节。
     
          int major = 0;
          int minor = 2;    // 小版本号已经修改为2
          int revision = 0;
          fwrite(&major, sizeof(int), 1, fp);
          fwrite(&minor, sizeof(int), 1, fp);
          fwrite(&revision, sizeof(int), 1, fp);
          fwrite(net->seen, sizeof(size_t), 1, fp);  // 这里的数据长度,在linux下64字节
          
          之前的版本是，fwrite(net.seen, sizeof(int), 1, fp); // 32字节
          
      finetune老版本的模型时，加载模型部分，做了兼容处理
      
      函数：load_weights_upto
        
        int major;  // 主版本号，保存模型时为0
        int minor;  // 次版本号，新版本时保存模型时已经修改为了2
        int revision;
        fread(&major, sizeof(int), 1, fp);
        fread(&minor, sizeof(int), 1, fp);
        fread(&revision, sizeof(int), 1, fp);
        if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){    
            fread(net->seen, sizeof(size_t), 1, fp);
        } else {                             // 兼容老版本训练的模型
            int iseen = 0;
            fread(&iseen, sizeof(int), 1, fp);   // 兼容老版本训练的模型，用int型获取数值
            *net->seen = iseen;
        }
        
        老版本，直接就是fread(net->seen, sizeof(int), 1, fp);
        
使用时的注意：
        解析模型读取数据时，要注意数据偏移的长度。
        例如，在将训练好的darknet的weights模型转为caffe时，转换工具，要注意更新。
