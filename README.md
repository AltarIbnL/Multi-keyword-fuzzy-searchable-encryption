# Multi-keyword-fuzzy-searchable-encryption
The Description of the project you can watch my blog

https://blog.csdn.net/weixin_39032619/article/details/110436737#comments_14140681

主要实现的论文是王恺璇的《面向多关键字的模糊密文搜索方法》

关键字集合只是单纯的通过计算5个案例文档中出现频率最高的10个单词；

因为论文中对布隆过滤器的位数有要求，所以代码中 BlommFilter 是从pybloom库 里面把源代码扒下来简单修改的（源码是自动根据输入的参数生成位数）

总的来说论文有些地方不是说的很清楚，所以实现的效果也一般般，只是简单的当做一个练手，本人的代码能力也不强，希望各位多多见谅。



