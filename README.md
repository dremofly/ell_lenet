.
├── pc
│   ├── cntk
│   │   ├── mnist.model
│   │   └── mnist.py
│   └── ell
│       ├── call_model.py
│       ├── model.ell
│       ├── testSample
│       │   └── img_27.jpg
│       └── tutorial_helpers.py
├── pi3
│   ├── call_model.py
│   ├── model.ell
│   ├── testSample
│   │   └── img_27.jpg
│   ├── tutorial_helpers.py
│   └── tutorial.py
└── readme.md

#步骤:

##进入pc/ell文件夹
```
cd pc/ell
```
##将model.ell转化成cmake project
```
python ~/ELL/ELL-master/tools/wrap/wrap.py model.ell -lang python -target host
```

##编译cmake project
```
cd host
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make && cd ../..
```

##测试模型
```
python call_model.py
```

##生成树莓派使用的模型
```
python ~/ELL/ELL-master/tools/wrap/wrap.py model.ell -lang python -target pi3
```

##将树莓派使用的模型复制到pi3文件夹
将整个pi3文件夹(包括刚刚复制过去的pi3)复制到树莓派里面

##在树莓派中进行编译,并运行(以下操作在树莓派中完成)
```
cd pi3
mkdir build
cd build 
cmake -DCMAKE_BUILD_TYPE=Release && make && cd ../..
python tutorial.py
```
