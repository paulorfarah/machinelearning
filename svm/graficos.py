# -*- encoding: iso-8859-1 -*-

import numpy as np
import matplotlib.pyplot as plt

# plotar acuracias comparação etre os kernels
ac_li  = [0.64,0.5933333333333334,0.6266666666666667,0.6466666666666666,0.62,0.6633333333333333,0.6166666666666667,0.63,0.6633333333333333,0.59,0.6333333333333333,0.59,0.5966666666666667,0.6533333333333333,0.6466666666666666,0.65,0.6133333333333333,0.61,0.5966666666666667,0.65,0.6033333333333334,0.6566666666666666,0.6533333333333333,0.6233333333333333,0.6733333333333333,0.6466666666666666,0.62,0.6366666666666667,0.61,0.6133333333333333]
ac_rbf = [1.0,0.9966666666666667,0.9966666666666667,1.0,0.9966666666666667,1.0,0.9933333333333333,0.9933333333333333,0.99,0.9866666666666667,1.0,0.9966666666666667,1.0,1.0,0.98,0.9933333333333333,0.9966666666666667,0.99,0.9966666666666667,0.9966666666666667,1.0,1.0,0.9933333333333333,1.0,0.9933333333333333,0.9966666666666667,1.0,1.0,1.0,0.9966666666666667]

ind = np.arange(30)  # the x locations for the groups
# width = 0.6  # the width of the bars
plt.plot(ac_li, marker='x', label='SVM Linear')
plt.plot(ac_rbf, marker='.', label='SVM RBF')

# plt.plot(acuracias_nb, marker='D', label='Naive Bayes')
# plt.plot(acuracias_lda, marker='*', label='LDA')
# plt.plot(acuracias_lr, marker='v', label='Logistic Regression')
plt.legend()
x = range(30)
plt.xticks(ind, x)

plt.savefig('resultados/acuracia_treino_svm_gaussian.pdf')

# plotar acuracias comparação etre os kernels
plt.close()
t_li  = [279,293,288,296,288,272,294,286,289,285,293,289,287,299,293,298,289,288,291,289,288,290,297,289,277,289,297,290,293,295]
t_rbf = [16,29,49,15,24,9,18,21,17,19,9,24,11,36,61,17,59,19,14,16,12,18,30,28,21,20,16,10,13,18]

ind = np.arange(30)  # the x locations for the groups
# width = 0.6  # the width of the bars
plt.plot(t_li, marker='x', label='SVM Linear')
plt.plot(t_rbf, marker='.', label='SVM RBF')

# plt.plot(acuracias_nb, marker='D', label='Naive Bayes')
# plt.plot(acuracias_lda, marker='*', label='LDA')
# plt.plot(acuracias_lr, marker='v', label='Logistic Regression')
plt.legend()
x = range(30)
plt.xticks(ind, x)

plt.savefig('resultados/vetores_treino_svm_gaussian.pdf')
plt.close()


### parte 2

acuracias = [0.9666, 0.9918, 0.9768, 0.9338, 0.9631, 0.9417]
ind = np.arange(len(acuracias))  # the x locations for the groups
width = 0.6       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, acuracias, width)
ax.set_ylabel('Acuracia')
ax.set_xlabel('Classificadores')

plt.ylim(0.9, 1)
ax.yaxis.grid(True)

#imprimir valores nas barras
for i, v in enumerate(acuracias):
    plt.text(i - 0.35, v + 0.002, " " + str(v),  va='center', fontweight='bold')

plt.xticks(ind, ('SVM Linear', 'SVM RBF', 'kNN', 'Naive Bayes', 'LDA', 'Log. Regression'))

# plt.show()
plt.savefig('acuracias_classificadores.pdf')
plt.close()

# rects2 = ax.bar(ind, women_means, width, bottom=men_means, color=(1.0,0.5,0.62))#'hotpink')

#tempo de busca: 1:12:16.019000

#tempo_treino = [0:00:03.877000,0:00:08.554000,0:00:00.153000,0:00:00.291000,0:00:00.928000,0:00:02.881000]

#tempo_execucao = [0:00:14.237000,1:12:29.620000,0:00:40.119000,0:00:01.062000,0:00:01.251000,0:00:03.193000]

vetores = [614, 194]
ind = np.arange(len(vetores))  # the x locations for the groups
# width = 0.2       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, vetores, width)
ax.set_ylabel('#vetores de suporte')
ax.set_xlabel('Classificadores')

# plt.ylim(0.9, 1)
ax.yaxis.grid(True)

#imprimir valores nas barras
for i, v in enumerate(vetores):
    plt.text(i-0.08, v + 10, " " + str(v),  va='center', fontweight='bold')

plt.xticks(ind, ('SVM Linear', 'SVM RBF'))

# plt.show()
plt.savefig('vetores_svm2.pdf')
plt.close()