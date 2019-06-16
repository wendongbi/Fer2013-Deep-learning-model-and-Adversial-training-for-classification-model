import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm


# plot acc
y1 = np.load('./test_acc_1.npy')
y2 = np.load('./test_acc_2.npy')
y3 = np.load('./test_acc_3.npy')
y4 = np.load('./test_acc_4.npy')
y5 = np.load('./test_acc_5.npy')
y6 = np.load('./test_acc_6.npy')
y7 = np.load('./test_acc_7.npy')
y8 = np.load('./test_acc_8.npy')
y9 = np.load('./test_acc_9.npy')
y10 = np.load('./test_acc_11.npy')
y11 = np.load('./test_acc_10.npy')

x = [i for i in range(len(y1[:200]))]
x = np.array(x)
print(y1.shape, y2.shape,x.shape)
plt.figure(figsize=(15, 15))
plt.subplot(211)
l1, = plt.plot(x, y1[:200])
l2, = plt.plot(x, y2[:200])
l3, = plt.plot(x, y3[:200])
l4, = plt.plot(x, y4[:200])
l5, = plt.plot(x, y5[:200])
l6, = plt.plot(x, y6[:200])
l7, = plt.plot(x, y7[:200])
l8, = plt.plot(x, y8[:200])
l9, = plt.plot(x, y9[:200])
l10, = plt.plot(x, y10[:200])
l11, = plt.plot(x, y11[:200])

plt.legend([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11], ['resnet18 rgb', 'resnet18 gray', 'Densenet121', 'dpn92', 'ResNeXt29_2x64d', 'preact_resnet18', 'vgg19', 'MobileNet', 'resnet101', 'resnet50', 'SENet18'],
    loc='lower right')
plt.title('Test accuracy')

# plot the validation accuracy
y1 = np.load('./val_acc_1.npy')
y2 = np.load('./val_acc_2.npy')
y3 = np.load('./val_acc_3.npy')
y4 = np.load('./val_acc_4.npy')
y5 = np.load('./val_acc_5.npy')
y6 = np.load('./val_acc_6.npy')
y7 = np.load('./val_acc_7.npy')
y8 = np.load('./val_acc_8.npy')
y9 = np.load('./val_acc_9.npy')
y10 = np.load('./val_acc_11.npy')
y11 = np.load('./val_acc_10.npy')

x = [i for i in range(len(y1[:200]))]
x = np.array(x)
print(y1.shape, y2.shape,x.shape)

plt.subplot(212)
l1, = plt.plot(x, y1[:200])
l2, = plt.plot(x, y2[:200])
l3, = plt.plot(x, y3[:200])
l4, = plt.plot(x, y4[:200])
l5, = plt.plot(x, y5[:200])
l6, = plt.plot(x, y6[:200])
l7, = plt.plot(x, y7[:200])
l8, = plt.plot(x, y8[:200])
l9, = plt.plot(x, y9[:200])
l10, = plt.plot(x, y10[:200])
l11, = plt.plot(x, y11[:200])

plt.legend([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11], ['resnet18 rgb', 'resnet18 gray', 'Densenet121', 'dpn92', 'ResNeXt29_2x64d', 'preact_resnet18', 'vgg19', 'MobileNet', 'resnet101', 'resnet50', 'SENet18'],
    loc='lower right')
plt.title('Validation accuracy')
plt.show()
