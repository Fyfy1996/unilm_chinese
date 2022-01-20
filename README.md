# unilm_chinese
A Unilm Chinese(pretraind by YunwenTech) Project base tested on couplet Generating
## model and code structure
use the structure from https://github.com/YunwenTechnology/Unilm
## data
Use the Chinese couplet dataset, more details can be seen in data folder
## Result
Examples:
    上联：  苦盼郎归，每唤登临，心中只恨东山矮
    Bot:   闲思子去，不堪怅望，眼底长留西子愁
    原下联：遥思雁去，频添怅慨，石上长留脚印深

    上联：  天降甘霖，华夏人民奔富路
    Bot:   地生春雨，神州百姓沐和风
    原下联：党行善政，崭新时代沐春风
    
## Environments and training
Trained on a single A100(40G) with epoch 3 on about 700k samples, costs about 7 hours.
