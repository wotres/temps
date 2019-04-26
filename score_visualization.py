# 파일 읽어옴
with open('models_score.txt', 'r') as f:
    lines = f.readlines()

emb_sz = []
layer = []
score = []
name = []

for i, line in enumerate(lines):
    paramString = line.split(' ')
    name.append(line)
    emb_sz.append(paramString[0].split('params_es')[1].split('_')[0])
    layer.append(paramString[0].split('layer')[1].split('_')[0])

    count = 0
    for j in range(len(paramString[1])):
        if paramString[1][j] == '*':
            count += 5
        elif paramString[1][j] == '/':
            count += 2.5

    score.append(count)

# for l in score:
#     print(l)


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.xaxis.set_ticks(np.arange(100,200, 50))
# ax.yaxis.set_ticks(np.arange(100,200, 50))

# for x, y, z, n in zip(emb_sz, layer, score, name):
#     print(x, y, z, n)
#     x = str(float(x)/20)
#     # y = str(float(y)*10)
#     print(x)
# #     print(word)
# #     ax = fig.add_subplot(111, projection='3d') 
#     # x, y, z = get_embeddings[i]
#     ax.scatter(x, y, z)
# #     ax.annotate(word, xyz=(x,y,z))
#     # ax.text(x,y,z,  '%s' % (str(n)), size=10, zorder=1, color='k') 
# #     plt.scatter(x,y,z)
# #     ax.scatter(x, y, z, c = z, s= 20, alpha=0.5, cmap=plt.cm.Greens)
# #     plt.annotate(word, xyz=(x,y,z))
#     # if(i == 100):
#     #     break;
# plt.show()

############################
import matplotlib.pyplot as plt

x = [0,1,2,3,4,5,100]
y = [10,20,15,18,7,19,1]
# xlabels = ['jan','feb','mar','apr','may','jun']

# xlabelsnew = []
# for i in xlabels:
#     if i not in ['feb','jun']:
#         i = ' '
#         xlabelsnew.append(i)
#     else:
#         xlabelsnew.append(i)
plt.scatter(layer, score)
# plt.plot(emb_sz,score)
# plt.xticks(range(4,10))
# plt.xticks(range(0,len(x)),xlabelsnew,rotation=45)
plt.show()