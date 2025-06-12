import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt


## Obtenção de uma política ótima para o problema de controle de estoques

T = 10                                 ##Número de estágios/épocas de decisão
n = 3                                  ##Capacidade do armazém de estoques
S = list(range(0,n+1))                 ##Espaço de estados
r = 4                                  ##Preço de venda do produto
c = 2                                  ##Custo de aquisição do produto
h = 0.5                                ##Custo de manter estoques

vetor_demandas = np.array([0,1,2])                      ##Valores possíveis da demanda
probs_demandas = np.array([1/4, 1/2, 1/4])              ##Dist. de probabilidades da demanda

##Função de transição de estado
def transicao(s,x,d):

    return np.maximum(s+x-d, 0)

##Função recompensa
def recompensa(s,x,d):

    return r*np.minimum(s+x,d)-c*x-h*np.maximum(s+x-d, 0)

# def custo_fixo(x):

#     if x == 0:
#         return 0
    
#     if x > 0:
#         return 10
       

##Lista com funções de valor para cada estágio
funcoes_valor_ótimas = np.zeros((T+1, len(S)))    ##Matriz cuja linha é a função de valor no tempo t
##Lista com políticas ótimas para cada estágio
pol_ótimas = np.zeros((T, len(S)), dtype="int")

##Cálculo no estágio final (não há decisão)
v_residual = np.zeros(len(S))

for s in S:

    v_residual[s] = -h*s


##Guarda o valor residual em funcoes_valor_ótimas
funcoes_valor_ótimas[T] = v_residual

##Loop do estágio T-1 até o estágio 0
for t in reversed(range(0,T)):

    v_otima_t = np.zeros(len(S))
    pol_otima = np.zeros(len(S))

    ##Loop de estados
    for s in S:

        ##Geração conjunto de decisões viáveis
        Xs = list(range(0, n-s+1))
        
        ##Valor da melhor decisão
        valor_melhor_decisao = -np.inf
        ##Loop do conjunto de decisões viáveis
        for x in Xs:

            ##Computa valor esperado da recompensa para o estado s e decisão x
            ##r_esperada = np.dot(recompensa(s,x,vetor_demandas), probs_demandas)

            ##Computa valor esperado da função de valor ótima para o próximo período
            ##v_proxima_esperado = 0

            valor_esperado_decisao = 0

            for j in range(len(vetor_demandas)):

                demanda = vetor_demandas[j]
                
                rt = recompensa(s,x,demanda)         ##Recompensa no tempo t
                s_proximo = transicao(s,x,demanda)   ##Estado s_t+1
                valor_s_proximo = funcoes_valor_ótimas[t+1,s_proximo]   ##Valor estado s_t+1

                valor_esperado_decisao+=(rt+valor_s_proximo)*probs_demandas[j]

            if valor_esperado_decisao > valor_melhor_decisao:

                valor_melhor_decisao = valor_esperado_decisao
                melhor_decisao = x
        
        v_otima_t[s] = valor_melhor_decisao
        pol_otima[s] = melhor_decisao

    funcoes_valor_ótimas[t] = v_otima_t
    pol_ótimas[t] = pol_otima

for t in range(0,T):

    print("Política ótima no tempo t = ", t, " = ", pol_ótimas[t])

for t in range(0,T+1):

    print("F Valor ótima em t = ", t, " = ", funcoes_valor_ótimas[t])



#Aplicação da política (simulação)

print("Simulação da política ótima")
rng = default_rng()    ##Gerador de números aleatórios

s = s0 = 0    ##Estado inicial

realizacao_estados = [s0]
realizacao_demandas = []
realizacao_decisoes = []


for t in range(T):

    ##Amostra a demanda no tempo t
    d = rng.choice(vetor_demandas, p = probs_demandas)

    print("Estado no tempo  t = ", t, " é ", d)

    realizacao_demandas.append(d)

    print("Demanda no tempo t = ", t, " é ", d)
    
    ##Toma decisão no tempo t em função do estado atual s
    x = pol_ótimas[t,s]

    realizacao_decisoes.append(x)

    print("Decisão no tempo t = ", t, " é ", x)

    ##Atualiza novo estado (função de transição de estados)
    s = transicao(s,x,d)

    ##Guarda novo estado
    realizacao_estados.append(s)


plt.plot(realizacao_estados,marker='.', label = "Nível estoque")
plt.plot(realizacao_demandas,marker='.', label = "Demanda")
plt.plot(realizacao_decisoes, ls="",marker='.', label = "Decisao")
plt.legend()