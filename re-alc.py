import os
import numpy as np
import sys
import random
sys.setrecursionlimit(2000)

#%%
#==============================================================================================
#Primero, tenemos las funciones hechas en los laboratorios

#ACLARACIÓN 
# Para correr ciertas funciones, y que sea mas rápido y práctico, reemplazamos, algunas funciones hechas por nosotros
# por las que nos da numpy que son más optimizadas. Para cumplir con el formato de entrega, las volvimos a la original.
# por ejemplo: "multiplicacion_mat", por "@"; 
#              "matriz_transpuesta", por ".T";

#%%
# Laboratorio 1

def error(x, y):
    """
    Recibe dos numeros x e y, y calcula el error de aproximar x usando y en float64.
    Si son vectores/matrices, devuelve la norma del error (Euclídea).
    """
    # caso 1: escalares (números sueltos)
    es_escalar_x = isinstance(x, (int, float, np.number))
    es_escalar_y = isinstance(y, (int, float, np.number))

    if es_escalar_x and es_escalar_y:
        return abs(x - y)
    
    # caso 2: vectores o matrices 
    # Error = (sum((x - y)^2))^(1/2)
    diff = np.array(x) - np.array(y)
    suma = 0
    for elem in (diff) :
        suma += (elem)**2
    return suma**(0.5)

def error_relativo(x, y):
    """
    Calcula el error relativo: |x-y|/|x|
    """
    # calculamos el error absoluto (reutilizamos la función anterior)
    err_abs = error(x, y)
    
    # calculamos la norma/valor absoluto de x
    if isinstance(x, (int, float, np.number)):
        norm_x = abs(x)
    else:
        # norma euclídea
        norm_x = 0
        for elem in x :
            norm_x += (elem)**2
        norm_x = (norm_x) ** 0.5
    # chequeo de seguridad para división por cero
    if norm_x == 0:
        if err_abs == 0:
            return 0.0
        return float('inf') # lanzar excepción

    return err_abs / norm_x

def matricesIguales(A, B):
    """
    Devuelve True si ambas matrices son iguales y False en otro caso.
    Considerar que las matrices pueden tener distintas dimensiones, ademas de distintos valores.
    """
    # chequeo de dimensiones
    if A.shape != B.shape:
        return False
    
    epsilon = 1e-08
    filas = A.shape[0] 
    
    # recorremos fila por fila con Slicing
    for i in range(filas):
        # agarramos la fila 'i' completa de A y de B
        fila_A = A[i, :] 
        fila_B = B[i, :]
        
        # restamos las filas 
        diferencia_fila = fila_A - fila_B
        
        # verificamos si algún elemento de la fila supera el error
        if any(abs(x) > epsilon for x in diferencia_fila):
            return False
            
    return True
#%%
# Test de Laboratorio 1
def sonIguales(x,y,atol=1e-08):
 return np.allclose(error(x,y),0,atol=atol)

assert(not sonIguales(1,1.1))
assert(sonIguales(1,1 + np.finfo('float64').eps))
assert(not sonIguales(1,1 + np.finfo('float32').eps))
assert(not sonIguales(np.float16(1),np.float16(1) + np.finfo('float32').eps))
assert(sonIguales(np.float16(1),np.float16(1) + np.finfo('float16').eps,atol=1e3))

assert(np.allclose(error_relativo(1,1.1), 0.1))
assert(np.allclose(error_relativo(2,1), 0.5))
assert np.allclose(error_relativo(-1, -1), 0)
assert np.allclose(error_relativo(1, -1), 2)

assert(matricesIguales(np.diag([1,1]),np.eye(2)))
assert(matricesIguales(np.linalg.inv(np.array([[1,2],[3,4]]))@np.array([[1,2],[3,4]]),np.eye(2)))
assert(not matricesIguales(np.array([[1,2],[3,4]]).T,np.array([[1,2],[3,4]])))

#%%
# Laboratorio 2

def rota(theta):
    """   
    Recibe un angulo theta y retorna una matriz de 2x2
    que rota un vector dado en un angulo theta
    """
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, -s],
        [s,  c]
    ])

def escala(s):
    """
    Recibe una tira de numeros s y retorna una matriz cuadrada de
    n x n , donde n es el tamano de s.
    La matriz escala la componente i de un vector de Rn en un factor s[i]
    """
    n = len(s)
    # creamos matriz de ceros
    M = np.zeros((n, n))
    
    # llenamos solo la diagonal principal 
    for i in range(n):
        M[i, i] = s[i]
        
    return M

def rota_y_escala(theta, s):
    """
    Recibe un angulo theta y una tira de numeros s,
    y retorna una matriz de 2x2 que rota el vector en un angulo theta
    y luego lo escala en un factor s
    """    
    # obtenemos R (rotación)
    R = rota(theta)
    
    # obtenemos S (escalado)
    # Como rota devuelve 2x2, asumimos que 's' tiene 2 elementos para que sea compatible
    S = escala(s) 
    
    # Orden: Primero Rota (derecha), Luego Escala (izquierda) -> S * R
    return S @ R

def afin(theta, s, b):
    """
    Recibe un angulo theta, una tira de numeros s (en R2), y un vector b en R2.
    Retorna una matriz de 3x3 que rota el vector en un angulo theta,
    luego lo escala en un factor s y por ultimo lo mueve en un valor fijo b
    """
    # obtenemos la parte lineal (2x2) usando la función anterior
    M_lineal = rota_y_escala(theta, s)
    
    # creamos la matriz de 3x3 vacía
    M_afin = np.zeros((3, 3))
    
    # rellenamos la parte de rotación y escala (arriba a la izquierda)
    M_afin[0:2, 0:2] = M_lineal
    
    # rellenamos la parte de traslación (columna 2, filas 0 y 1)
    # b tiene que entrar como columna. Si b es array plano, numpy lo acomoda, 
    # pero a veces es mejor ser explícito. Asumimos b vector de 2 elementos.
    M_afin[0, 2] = b[0]
    M_afin[1, 2] = b[1]
    
    # el 1 de la coordenada homogénea (abajo a la derecha)
    M_afin[2, 2] = 1.0
    
    return M_afin

def trans_afin(v, theta, s, b):
    """
    Recibe un vector v (en R2), un angulo theta,
    una tira de numeros s (en R2), y un vector b en R2.
    Retorna el vector w resultante de aplicar la transformacion afin a v
    """
    # conseguimos la matriz completa 3x3
    M = afin(theta, s, b)
    
    # convertimos v a homogéneo [x, y, 1] manualmente
    v_homogeneo = np.array([v[0], v[1], 1.0])
    
    # multiplicamos Matriz x Vector
    w_homogeneo = M @ v_homogeneo
    
    # devolvemos solo las coordenadas cartesianas (x, y)
    return w_homogeneo[0:2]

#%%
# Test Laboratorio 2

# Tests para rota
assert(np.allclose(rota(0),np.eye(2)))
assert(np.allclose(rota(np.pi/2), np.array([[0,-1],[1,0]])))
assert(np.allclose(rota(np.pi),np.array([[-1,0],[0,-1]])))

# Test para escala
assert(np.allclose(escala([2,3]),np.array([[2,0],[0,3]])))
assert(np.allclose(escala([1,1,1]),np.eye(3)))
assert(np.allclose(escala([0.5,0.25]),np.array([[0.5,0],[0,0.25]])))

# Test para rota y escala
assert(np.allclose(rota_y_escala(0,[2,3]),np.array([[2,0],[0,3]])))
assert(np.allclose(rota_y_escala(np.pi/2,[1,1]),np.array([[0,-1],[1,0]])))
assert(np.allclose(rota_y_escala(np.pi,[2,2]),np.array([[-2,0],[0,-2]])))

# Test para afin
assert(np.allclose(afin(0,[1,1],[1,2]),np.array([[1,0,1],[0,1,2],[0,0,1]])))
assert(np.allclose(afin(np.pi/2,[1,1],[0,0]),np.array([[0,-1,0],[1,0,0],[0,0,1]])))
assert(np.allclose(afin(0,[2,3],[1,1]),np.array([[2,0,1],[0,3,1],[0,0,1]])))

# Tests para trans afin
assert(np.allclose(trans_afin(np.array([1,0]),np.pi/2,[1,1],[0,0]),np.array([0,1])))
assert(np.allclose(trans_afin(np.array([1,1]),0,[2,3],[0,0]),np.array([2,3])))
assert(np.allclose(trans_afin(np.array([1,0]),np.pi/2,[3,2],[4,5]),np.array([4,7])))
#%%
# Laboratorio 3

def norma(x, p):
    """
    Devuelve la norma p del vector x.
    Soporta p = 'inf' (infinito).
    """
    # si es un solo número
    if isinstance(x, (int, float, np.number)):
        return abs(x)
        
    # caso norma Infinito: max|x_i|
    if p == 'inf':
        # Usamos max y abs de python nativo sobre el array
        return max(abs(xi) for xi in x)
    
    # caso norma p general: (sum |x_i|^p)^(1/p)
    suma = sum(abs(xi)**p for xi in x)
    return suma**(1/p)

def normaliza(X, p):
    """
    Recibe X, una lista de vectores no vacios, y un escalar p.
    Devuelve una lista donde cada elemento corresponde a normalizar 
    los elementos de X con la norma p.
    """
    Y = []
    # iteramos sobre cada vector en la lista X
    for v in X:
        nrm = norma(v, p)
        # evitamos división por cero 
        if nrm == 0:
            Y.append(v) # O un vector de ceros
        else:
            Y.append(v / nrm)
    return Y

def normaMatMC(A, q, p, Np):
    """
    Devuelve la norma ||A||_{q,p} aproximada y el vector x donde se alcanza el maximo,
    usando el metodo de Monte Carlo con Np iteraciones.
    """
    # aseguramos que A sea array para poder multiplicar
    A = np.array(A, dtype=float)
    n = A.shape[1]

    maximo = -1.0 # la norma siempre es positiva
    vector_max = None

    for _ in range(Np):
        # generación de vector aleatorio 
        
        #caso 1: Norma 1 -> Vector canónico aleatorio (un 1, resto 0s)
        if p == 1:
            x = np.zeros(n)
            idx = random.randint(0, n - 1) # Entero aleatorio entre 0 y n-1
            x[idx] = 1.0
            
        #caso 2: Norma Inf -> Vector de +1 y -1
        elif p == 'inf':
            # List comprehension con random.choice
            lista_random = [random.choice([-1, 1]) for _ in range(n)]
            x = np.array(lista_random, dtype=float)
            
        #caso 3: Norma 2 u otra -> Gaussiana normalizada
        else:
            # random.gauss(mu, sigma) es nativo de python
            lista_random = [random.random() - 0.5 for _ in range(n)] # Uniforme centrada en 0
            # (Nota: Para p=2 idealmente usariamos random.gauss(0,1), 
            # pero uniforme funciona para llenar el espacio en MC básico)
            x_raw = np.array(lista_random, dtype=float)
            # Normalizamos para que ||x||_p = 1
            nrm_x = norma(x_raw, p)
            if nrm_x == 0: continue # Skip si salió vector nulo (rarísimo)
            x = x_raw / nrm_x

        # Cálculo del cociente de Rayleigh generalizado
        # como x ya está normalizado (||x||_p = 1), solo calculamos ||Ax||_q
        vector_transformado = A @ x
        norma_actual = norma(vector_transformado, q)

        if norma_actual > maximo:
            maximo = norma_actual
            vector_max = x

    return maximo, vector_max

def normaExacta(A, p=[1, 'inf']):
    """
    Devuelve una lista con las normas 1 e infinito de una matriz A 
    usando las expresiones del enunciado.
    """
    # A debe ser array para usar shape
    if not isinstance(A, np.ndarray):
        A = np.array(A)
        
    filas, columnas = A.shape
    
    # Norma 1: Máxima suma absoluta de COLUMNAS 
    # ||A||_1 = max_j (sum_i |a_ij|)
    max_col = 0.0
    for j in range(columnas):
        # sumamos el valor absoluto de la columna j
        # slicing A[:, j] es la columna j
        suma_actual = np.sum(np.abs(A[:, j]))
        if suma_actual > max_col:
            max_col = suma_actual
            
    # Norma Infinito: Máxima suma absoluta de FILAS 
    # ||A||_inf = max_i (sum_j |a_ij|)
    max_fila = 0.0
    for i in range(filas):
        # slicing A[i, :] es la fila i
        suma_actual = np.sum(np.abs(A[i, :]))
        if suma_actual > max_fila:
            max_fila = suma_actual

    if p == [1, 'inf']:
        return [max_col, max_fila]
    else:
        return None
            
    return [max_col, max_fila]

def condMC(A, p, Np=1000):
    """
    Devuelve el numero de condicion de A usando la norma inducida p (Monte Carlo).
    """
    # calculamos norma de A
    n_A, _ = normaMatMC(A, p, p, Np)
    
    # calculamos inversa
    inv_A = inversa_gauss_jordan(A)
    
    # calculamos norma de la inversa
    n_invA, _ = normaMatMC(inv_A, p, p, Np)
    
    return n_A * n_invA

def condExacto(A, p):
    """
    Devuelve el numero de condicion de A usando formulas exactas (para p=1 o p=inf).
    """
    # las normas exactas devuelve una lista [norma1, normaInf]
    normas_A = normaExacta(A)
    
    inv_A = inversa_gauss_jordan(A)
    normas_invA = normaExacta(inv_A)
    
    if p == 1:
        # posicion 0 de la lista
        return normas_A[0] * normas_invA[0]
    elif p == 'inf':
        # posicion 1 de la lista
        return normas_A[1] * normas_invA[1]
    else:
        return None
    
# función inversa con eliminación gaussiana para calcular la condicion de las matrices.

def inversa_gauss_jordan(A):
    """
    Calcula la inversa de A usando Gauss-Jordan
    """
    n = A.shape[0]
    
    # Creamos una matriz vacía de n filas y 2n columnas. Matriz A ampliada a la identidad.
    # [ A | Identidad ]
    M = np.zeros((n, 2 * n), dtype=float)
    
    # llenamos la parte izquierda con A
    M[:, :n] = A
    
    # llenamos la parte derecha con la Identidad (manualmente para no usar np.eye)
    M[:, n:]= np.eye(n)
        
    # 2. ELIMINACIÓN GAUSSIANA
    for i in range(n):
        # paso 1: Pivoteo (Buscar el mayor valor en la columna i) 
        # miramos desde la fila i hacia abajo y busco el indice máximo
        idx_max = np.argmax(np.abs(M[i:, i])) + i
        pivot = M[idx_max, i]
        
        # chequeo de singularidad (si el pivote es casi 0, no hay inversa)
        if abs(pivot) < 1e-12:
            return None 
            
        # intercambio de filas si es necesario
        if idx_max != i:
            M[[i, idx_max]] = M[[idx_max, i]]
            
        # Paso B: Normalizar la fila pivote
        # dividimos toda la fila por el pivote para que nos quede un 1 en la diagonal
        # operamos sobre la fila completa (vectorizado)
        M[i, :] = M[i, :] / M[i, i]
        
        # Paso C: Hacer ceros en la columna 
        # restamos a todas las OTRAS filas (j != i)
        for j in range(n):
            if i != j:
                # calculamos el factor de anulación
                factor = M[j, i]
                
                # operación vectorial: Fila_j = Fila_j - factor * Fila_i
                M[j, :] -= factor * M[i, :]

    # extracción del res
    # devolvemos la mitad derecha de la matriz (columnas n hasta el final)
    return M[:, n:].copy()
    
#%%
# Test Laboratorio 3

# Tests norma
print("TESTS NORMA")
assert(np.allclose(norma(np.array([0,0,0,0]),1), 0))
assert(np.allclose(norma(np.array([4,3,-100,-41,0]),"inf"), 100))
assert(np.allclose(norma(np.array([1,1]),2),np.sqrt(2)))
assert(np.allclose(norma(np.array([1]*10),2),np.sqrt(10)))
assert(norma(np.random.rand(10),2)<=np.sqrt(10))
assert(norma(np.random.rand(10),2)>=0)

print("------ÉXITO!!!!\n")

# Tests normaliza
print("TEST NORMALIZA")

# caso borde
# print("---TEST NORMALIZA NULO")
# test_borde = normaliza([np.array([0,0,0,0])],2)
# assert(len(test_borde) == 1)
# assert(np.allclose(test_borde[0],np.array([0,0,0,0])))
# print("------ÉXITO!!!!")

# normaliza norma 2
print("---TEST NORMALIZA 2")
test_n2 = normaliza([np.array([1]*k) for k in range(1,11)],2)
assert(len(test_n2) != 0)
for x in test_n2:
    assert(np.allclose(norma(x,2),1))
print("------ÉXITO!!!!")

# normaliza norma 1
print("---TEST NORMALIZA 1")
test_n1 = normaliza([np.array([1]*k) for k in range(2,11)],1)
assert(len(test_n1) != 0)
for x in test_n1:
    assert(np.allclose(norma(x,1),1))
print("------ÉXITO!!!!")

# normaliza norma inf
print("---TEST NORMALIZA INF")
test_nInf = normaliza([np.random.rand(k) for k in range(1,11)],'inf')
assert(len(test_nInf) != 0)
for x in test_nInf:
    assert(np.allclose(norma(x,'inf'),1))

print("------ÉXITO!!!!\n")

# Tests normaExacta
print("TEST normaExacta")

assert(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]))[0],2))
assert(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]))[1],2))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]))[0] ,6))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]))[1],7))
assert(normaExacta(np.array([[1,-2],[-3,-4]]),2) is None)
assert(normaExacta(np.random.random((10,10)))[0] <=10)
assert(normaExacta(np.random.random((4,4)))[1] <=4)

print("------ÉXITO!!!!\n")

# Test normaMatMC
print("TEST normaMatMC")

nMC = normaMatMC(A=np.eye(2),q=2,p=1,Np=100000)
assert(np.allclose(nMC[0],1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),0,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),0,atol=1e-3))

nMC = normaMatMC(A=np.eye(2),q=2,p='inf',Np=100000)
assert(np.allclose(nMC[0],np.sqrt(2),atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) and np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))

A = np.array([[1,2],[3,4]])
nMC = normaMatMC(A=A,q='inf',p='inf',Np=1000000)
assert(np.allclose(nMC[0],normaExacta(A)[1],rtol=1e-1)) 

print("------ÉXITO!!!!\n")

# Test condMC
print("TEST condMC")

A = np.array([[1,1],[0,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaMatMC(A,2,2,10000)
normaA_ = normaMatMC(A_,2,2,10000)
condA = condMC(A,2,)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-2))

A = np.array([[3,2],[4,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaMatMC(A,2,2,10000)
normaA_ = normaMatMC(A_,2,2,10000)
condA = condMC(A,2)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-2))

print("------ÉXITO!!!!\n")

# Test condExacta
print("TEST condExacta")

A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaExacta(A)[0]
normaA_ = normaExacta(A_)[0]
condA = condExacto(A,1)
assert(np.allclose(normaA*normaA_,condA))

A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaExacta(A)[1]
normaA_ = normaExacta(A_)[1]
condA = condExacto(A,'inf')
assert(np.allclose(normaA*normaA_,condA))

print("------ÉXITO!!!!\n")

print("---FINALIZADO LABO 3!---")

#%%
# Laboratorio 4

# defino la funcion para emplear multiplicaciones entre matrices
def multiplicacion_mat(A, B):
    """Multiplicación matricial manual (C = A @ B)"""
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)

    # Si viene un vector suelto (ej: shape=(5,)), lo convertimos a matriz (5,1)
    # para que tenga filas y columnas y no rompa el código de abajo.
    if A.ndim == 1: A = A.reshape(1, -1)
    if B.ndim == 1: B = B.reshape(-1, 1)

    n_filas_A, n_cols_A = A.shape
    n_filas_B, n_cols_B = B.shape
    
    if n_cols_A != n_filas_B: return None
    
    C = np.zeros((n_filas_A, n_cols_B))
    
    for i in range(n_filas_A):
        for j in range(n_cols_B):
            # Tu lógica original (está perfecta)
            C[i, j] = np.sum(A[i, :] * B[:, j])
            
    return C

def calculaLU(A):
    """
    Calcula la factorización LU de A.
    Retorna (L, U, cant_op) o (None, None, cant_op) si no es posible.
    """
    
    if A is None:
        return None, None, 0
    # copiamos A para no modificar la original (float para evitar truncamiento)
    Ac = np.array(A, dtype=float) 
    n = Ac.shape[0]
    cant_op = 0
    
    if Ac.shape[0] != Ac.shape[1]:
        # Si no es cuadrada, no se puede hacer LU estándar
        return None, None, 0
    
    # inicializamos L con identidad (ceros y unos en diagonal)
    L = np.eye(n)

    # Algoritmo de eliminación 
    for i in range(n):
        # Verificación de Pivote 
        pivot = Ac[i][i]
        
        # Si el pivote es 0, no se puede descomponer sin permutar filas
        if abs(pivot) < 1e-12:
            return None, None, 0

        # Eliminación
        for j in range(i + 1, n): # recorro todas las filas j que están debajo de la fila actual i
            # Calculamos el factor (multiplicador)
            factor = Ac[j][i] / pivot
            L[j][i] = factor 
            cant_op += 1 # División
            
            # Restamos la fila: Fila_j = Fila_j - factor * Fila_i
            # Usamos slicing que es básico y eficiente
            # Ac[j, i:] son los elementos desde la columna i en adelante
            # no restamos la fila entera desde el principio (0). Restamos desde la 
            # columna i hasta el final por que las columnas anteriores a i ya son ceros
            # (debido a que ya pasamos por ahí en iteraciones anteriores)
            Ac[j, i:] = Ac[j, i:] - factor * Ac[i, i:]
            
            # Conteo ops: (resta + mult) * cantidad de elementos restantes 
            # --> dos operaciones por cada elemento restante
            elems_restantes = n - (i + 1)
            cant_op += 2 * elems_restantes

    # Lo que queda en Ac es U
    U = Ac
    return L, U, cant_op

def res_tri(M, b, inferior=True):
    """
    Resuelve Mx=b. Sirve para L (inferior=True) y U (inferior=False).
    """
    n = len(M)
    x = np.zeros(n) # vector solución
    
    # validamos dimensiones
    if len(b) != n:
        raise ValueError("Dimensiones incompatibles")

    if inferior:
        # sustitución hacia ADELANTE (L)
        for i in range(n):
            if abs(M[i][i]) < 1e-12: 
                raise ValueError("Matriz singular")
            
            # sumamos M[i][j] * x[j] para todas las columnas anteriores (j < i)
            suma = 0.0
            for j in range(i):
                suma += M[i][j] * x[j]
            
            x[i] = (b[i] - suma) / M[i][i]
            
    else:
        # --- Sustitución hacia ATRÁS (U) ---
        for i in range(n - 1, -1, -1):
            if abs(M[i][i]) < 1e-12: 
                raise ValueError("Matriz singular")
            
            # Sumamos M[i][j] * x[j] para todas las columnas siguientes (j > i)
            suma = 0.0
            for j in range(i + 1, n):
                suma += M[i][j] * x[j]
            
            x[i] = (b[i] - suma) / M[i][i]

    return x
"""
La implementación de la inversa usa factorización LU sin pivoteo.
Esto funciona para matrices con pivotes no nulos, pero puede fallar para matrices invertibles que requieren permutación de filas.
Para una versión general debería implementarse LU con pivoteo parcial (PA=LU).
"""
""" implementación anterior que usa la factorización LU sin pivoteo
def inversa(A):
    
    # Calcula la inversa empleando factorización LU.
    
    n = len(A)
    
    # descomponemos
    L, U, _ = calculaLU(A)
    
    # si calculaLU falló (pivote 0), la matriz no tiene inversa
    if L is None:
        return None

    # inicializamos la futura inversa
    inv_A = np.zeros((n, n))

    # resolvemos columna por columna
    for j in range(n):
        
        # Creamos la columna j de la Identidad

        e_j = np.zeros(n)
        e_j[j] = 1.0

        # resolvemos A * col_j = e_j usando LU
        # A = L*U. Entonces L*U*x = e_j
        
        # primero resolvemos L * y = e_j 
        y = res_tri(L, e_j, inferior=True)
        
        # ahora resolvemos U * x = y 
        # Este 'x' que encontramos ES la columna j de la inversa
        x_j = res_tri(U, y, inferior=False)

        # finalmente guardamos la columna encontrada en su lugar
        inv_A[:, j] = x_j

    return inv_A
"""

def calculaPA_LU(A):
    """
    Realiza la factorización PA = LU con pivoteo.
    Retorna (L, U, P, cant_op).
    """
    if A is None: return None, None, None, 0
    
    # ttabajamos con copias float
    U = np.array(A, dtype=float)
    n = U.shape[0]
    L = np.eye(n)
    P = np.eye(n) 
    cant_op = 0

    if U.shape[0] != U.shape[1]: return None, None, None, 0

    for i in range(n):
        # Buscamos el mayor valor absoluto en la columna i, desde la fila i para abajo
        pivot_idx = i
        max_val = abs(U[i, i])

        for k in range(i + 1, n):
            val_k = abs(U[k, i])
            if val_k > max_val:
                max_val = val_k
                pivot_idx = k
        
        pivot_val = U[pivot_idx, i]

        # Si el pivote máximo es casi 0, la matriz no tiene inversa
        if abs(pivot_val) < 1e-12:
            return None, None, None, 0

        # Si el mejor no está en la posición actual, intercambiamos (Swap)
        if pivot_idx != i:
            # Swap en U
            U[[i, pivot_idx], :] = U[[pivot_idx, i], :]
            # Swap en P
            P[[i, pivot_idx], :] = P[[pivot_idx, i], :]
            # Swap en L (solo la parte triangular inferior ya calculada)
            if i > 0:
                L[[i, pivot_idx], :i] = L[[pivot_idx, i], :i]

        # ELIMINACIÓN 
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            cant_op += 1

            # Restamos filas: Fila_j = Fila_j - factor * Fila_i
            # Slicing desde i para eficiencia (evita restar ceros anteriores)
            U[j, i:] = U[j, i:] - factor * U[i, i:]
            
            elems = n - i 
            cant_op += 2 * elems

    return L, U, P, cant_op

def inversa(A):
    """
    Calcula la inversa usando PA=LU para mayor robustez numérica.
    """
    n = len(A)
    # llamamos a la nueva función que devuelve P también
    L, U, P, _ = calculaPA_LU(A)
    
    if L is None: return None

    inv_A = np.zeros((n, n))

    for j in range(n):
        # vect canónico e_j
        e_j = np.zeros(n)
        e_j[j] = 1.0
        
        # sistema original: A x = e_j
        # multiplicamos por P: P A x = P e_j
        # como P A = L U, queda: L U x = P e_j
        
        # Aplicamos la permutación al vector b
        # Esto es multiplicar la matriz P por el vector e_j
        b_perm = multiplicacion_mat(P, e_j) 

        # resolvemos L y = b_perm
        y = res_tri(L, b_perm, inferior=True)
        
        # ahora U x = y
        x_j = res_tri(U, y, inferior=False)

        # guardamos la columna
        inv_A[:, j] = x_j

    return inv_A

def calculaLDV(A):
    """
    Factorización A = L D V.
    L: Triangular inferior (diag 1)
    D: Diagonal
    V: Triangular superior (diag 1)
    """
    # aplicamos LU que ya tenemos hecho
    L, U, ops = calculaLU(A)
    
    if L is None:
        return None, None, None, ops
        
    n = len(A)
    D = np.zeros((n, n))
    V = np.zeros((n, n))
    
    # separamos U en D y V
    # sabemos que U = D * V.
    # sos elementos de la diagonal de D son la diagonal de U.
    for i in range(n):
        diag_val = U[i][i]
        
        # Si la diagonal es 0, no podemos hacer LDV único (o D sería 0)
        if abs(diag_val) < 1e-12:
             return None, None, None, ops
             
        D[i][i] = diag_val
        
        # V se obtiene dividiendo cada fila de U por su diagonal
        # V[i, :] = U[i, :] / D[i][i]
        # Esto hace que la diagonal de V quede en 1.0
        V[i, :] = U[i, :] / diag_val
        
        # Sumamos operaciones (n divisiones por fila)
        ops += n
        
    return L, D, V, ops

def esSDP(A, atol=1e-8):
    """
    Chequea si es Simétrica Definida Positiva usando LDV.
    """
    n = A.shape[0]
    
    # verificar SIMETRÍA 
    # Recorremos solo el triángulo superior sin diagonal
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > atol:
                return False # No es simétrica
            
    #puedo optimizar esta parte del código con algunas funciones de numpy:
    #if np.max(np.abs(A - A.T)) > 1e-8:
    #    raise ValueError("La matriz no es simétrica. Cholesky requiere simetría.")
                
    # verificar DEFINIDA POSITIVA
    # Teorema: Una matriz Simétrica es Def. Positiva sii en su descomposición
    # LDL^T (o LDV), todos los elementos de D son > 0.
    
    L, D, V, _ = calculaLDV(A)
    
    if L is None: 
        return False # Si no se puede descomponer, no es SDP
        
    # Chequeamos que la diagonal de D sea toda positiva
    for i in range(n):
        if D[i][i] <= atol: # Tiene que ser estrictamente positivo (> 0)
            return False
            
    return True

#%%
# Test Laboratorio 4

# TESTS LU
print("TESTS calculaLU")

L0 = np.array([[1,0,0],
               [0,1,0],
               [1,1,1]])

U0 = np.array([[10,1,0],
               [0,2,1],
               [0,0,1]])

A =  L0 @ U0
L,U,nops = calculaLU(A)
assert(np.allclose(L,L0))
assert(np.allclose(U,U0))


L0 = np.array([[1,0,0],
               [1,1.001,0],
               [1,1,1]])

U0 = np.array([[1,1,1],
               [0,1,1],
               [0,0,1]])
A =  L0 @ U0
L,U,nops = calculaLU(A)
assert(not np.allclose(L,L0))
assert(not np.allclose(U,U0))
assert(np.allclose(L,L0,atol=1e-3))
assert(np.allclose(U,U0,atol=1e-3))
assert(nops == 13)

L0 = np.array([[1,0,0],
               [1,1,0],
               [1,1,1]])

U0 = np.array([[1,1,1],
               [0,0,1],
               [0,0,1]])

A =  L0 @ U0
L,U,nops = calculaLU(A)
assert(L is None)
assert(U is None)
assert(nops == 0)

assert(calculaLU(None) == (None, None, 0))

assert(calculaLU(np.array([[1,2,3],[4,5,6]])) == (None, None, 0))

print("-----ÉXITO!!!!\n")


## TESTS res_tri
print("TESTS res_tri")

A = np.array([[1,0,0],
              [1,1,0],
              [1,1,1]])

b = np.array([1,1,1])
assert(np.allclose(res_tri(A,b),np.array([1,0,0])))

b = np.array([0,1,0])
assert(np.allclose(res_tri(A,b),np.array([0,1,-1])))

b = np.array([-1,1,-1])
assert(np.allclose(res_tri(A,b),np.array([-1,2,-2])))

b = np.array([-1,1,-1])
assert(np.allclose(res_tri(A,b,inferior=False),np.array([-1,1,-1])))

A = np.array([[3,2,1],[0,2,1],[0,0,1]])
b = np.array([3,2,1])
assert(np.allclose(res_tri(A,b,inferior=False),np.array([1/3,1/2,1])))

A = np.array([[1,-1,1],[0,1,-1],[0,0,1]])
b = np.array([1,0,1])
assert(np.allclose(res_tri(A,b,inferior=False),np.array([1,1,1])))
print("-----ÉXITO!!!!\n")


# Test inversa
print("TESTS inversa")

def esSingular(A):
    try:
        np.linalg.inv(A)
        return False
    except:
        return True

# Por que no siempre es invertible, hacemos varios tests
ntest = 10
for i in range(ntest):
    A = np.random.random((4,4))
    A_ = inversa(A)
    if not esSingular(A):
        inversaConNumpy = np.linalg.inv(A)
        assert(A_ is not None)
        assert(np.allclose(inversaConNumpy,A_))
    else: 
        assert(A_ is None)

# Matriz singular devería devolver None
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
assert(inversa(A) is None)

print("-----ÉXITO!!!!\n")



# Test LDV:
print("TESTS calculaLDV")

L0 = np.array([[1,0,0],[1,1.,0],[1,1,1]])
D0 = np.diag([1,2,3])
V0 = np.array([[1,1,1],[0,1,1],[0,0,1]])
A =  L0 @ D0 @ V0
L,D,V, _ = calculaLDV(A)
assert(np.allclose(L,L0))
assert(np.allclose(D,D0))
assert(np.allclose(V,V0))


L0 = np.array([[1,0,0],[1,1.001,0],[1,1,1]])
D0 = np.diag([3,2,1])
V0 = np.array([[1,1,1],[0,1,1],[0,0,1.001]])
A =  L0 @ D0  @ V0
L,D,V, _ = calculaLDV(A)
assert(np.allclose(L,L0,1e-3))
assert(np.allclose(D,D0,1e-3))
assert(np.allclose(V,V0,1e-3))

print("-----ÉXITO!!!!\n")

# TESTS SDP
print("TESTS esSDP")

L0 = np.array([[1,0,0],[1,1,0],[1,1,1]])
D0 = np.diag([1,1,1])
A = L0 @ D0 @ L0.T
assert(esSDP(A))

D0 = np.diag([1,-1,1])
A = L0 @ D0 @ L0.T
assert(not esSDP(A))

D0 = np.diag([1,1,1e-16])
A = L0 @ D0 @ L0.T
assert(not esSDP(A))

L0 = np.array([[1,0,0],
               [1,1,0],
               [1,1,1]])
D0 = np.diag([1,1,1])
V0 = np.array([[1,0,0],
               [1,1,0],
               [1,1+1e-3,1]]).T
A = L0 @ D0 @ V0
assert(esSDP(A,1e-3))

print("-----ÉXITO!!!!\n")
print("---FINALIZADO LABO 4!---")



#%%
# Laboratorios 5

def producto_punto(u, v):
    """
    Producto escalar entre dos vectores u y v.
    Ambos deben tener la misma longitud.
    """
    if len(u) != len(v):
        raise ValueError("Vectores de distinta dimensión")

    suma = 0.0
    for i in range(len(u)):
        suma += float(u[i]) * float(v[i])

    return suma


def QR_con_GS(A, tol=1e-12):
    A = np.array(A, dtype=float)
    m, n = A.shape

    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()

        for k in range(j):
            R[k, j] = producto_punto(Q[:, k], v)
            v = v - R[k, j] * Q[:, k]

        R[j, j] = norma(v,2)

        if R[j, j] < tol:
            Q[:, j] = np.zeros(m)
        else:
            Q[:, j] = v / R[j, j]

    return Q, R

def QR_con_HH(A, tol=1e-12):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    """
    
    """
    Queremos factorizar A=QR usando reflectores de Householder.
    Un reflector tiene la forma: H=I−2uu^t con u unitario.
    En lugar de construir H que es muy caro, usamos:
    Hx=x−2u(u^tx)
    
    IDEA:
    En el método QR con Householder, no se construyen explícitamente los reflectores
    H = I − 2uu^t. En cada paso se aplica el reflector únicamente a una columna o a la
    submatriz activa, actualizando R como
    R ← R − 2u(u^tR)
    donde u^tR es un vector fila y no una matriz completa, lo que permite pagar un
    costo cuadrático por iteración. Los vectores uₖ se almacenan y la matriz Q se
    construye al final como el producto de los reflectores
    Q = H0*H1* ... H(n-1), de modo que el costo cúbico se paga una sola vez.
    """
    # converción a array float y copiamos para no romper el original
    # (valido primero si A es None o dimensiones como en LU)
    if A is None: return None, None
    R = np.array(A, dtype=float)
    
    m, n = R.shape
    
    #if m < n :
     #   return None, None
    # comento esta parte para usar la funcion en el tp con matrices no cuadradas
    
    # empezamos Q como una identidad
    Q = np.eye(m)

    # Iteramos sobre las columnas (k es el paso, el pivote)
    for k in range(min(m-1, n)):
        # extraemos el vector x (desde la diagonal para abajo)
        x = R[k:, k]
        
        # calculamos norma
        nrm_x = norma(x, 2)
        
        # si el vector ya es casi cero, saltamos
        if nrm_x < tol:
            continue

        # hacemos el vector 'u'
        # u = x +/- ||x|| * e1
        # elegimos el signo para evitar restas que den cero 
        signo = 1.0
        if x[0] < 0:
            signo = -1.0
            
        u = x.copy()
        u[0] += signo * nrm_x # Sumamos la norma al primer elemento
        
        # normalizamos u (u = u / ||u||)
        nrm_u = norma(u, 2)
        if nrm_u < tol: continue
        
        # dividimos manualmente cada elemento
        for i in range(len(u)):
            u[i] = u[i] / nrm_u
            
        # aplico H A R (por izquierda): R = R - 2u(u^t R) 
        # solo trabajamos en la submatriz R[k:, k:]
        # u tiene longitud (m-k). R_sub tiene (m-k) filas y (n-k) columnas.
        
        # recorremos todas las columnas 'j' que están a la derecha de donde estamos
        for j in range(k, n):
            # producto punto entre u y la columna actual de R
            # dot_val = u . R_columna_j
            columna_actual = R[k:, j]
            proyeccionr = producto_punto(u, columna_actual)
            
            # Actualizamos la columna: col = col - 2 * dot_val * u
            # Lo hacemos elemento a elemento (simulando resta de vectores)
            factor = 2.0 * proyeccionr
            for i in range(len(u)):
                R[k + i, j] = R[k + i, j] - factor * u[i]

        #### aplico H A Q (por derecha): Q = Q - 2(Q u)u^t 
        # Q se actualiza completo (todas las filas). 
        # el vector u solo afecta las ultimas columnas de Q (desde k).
        
        # recorro las  filas de Q
        for i in range(m):
            # producto punto entre la parte relevante de la fila i de Q y u
            # dot_val = Q[i, k:] . u
            fila_relevante_Q = Q[i, k:]
            proyeccionq = producto_punto(fila_relevante_Q, u)
            
            #actualización: fila = fila - 2 * dot_val * u^T
            factor = 2.0 * proyeccionq
            for j in range(len(u)):
                Q[i, k + j] = Q[i, k + j] - factor * u[j]

    return Q, R

# Comentario sobre complejidad:
# El algoritmo es cuadrático en cada iteración (O(n^2) por reflector: El costo por 
# iteración es cuadrático porque el reflector se aplica como un producto vector–matriz
# y no como una multiplicación entre matrices completas.). 
# Si construyera la matriz H explícita, sería cúbico por iteración y de orden 4 en total. 
# Al aplicar el reflector usando vectores y bucles sobre la submatriz, bajé la
# complejidad al mínimo posible para matrices densas, que es O(n^3) en total."

def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R    
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """
    if metodo == 'GS' :
        return QR_con_GS(A, tol)
    elif metodo == 'RH' :
        return QR_con_HH(A, tol)
    else:
        return None
    
# funciones para QR reducida
    
def QR_HH_reducida(A, tol=1e-12):
    """
    Calcula QR con Householder y devuelve la versión reducida.
    Q será de m x n
    R será de n x n
    """
    # llamamos a la versión FULL 
    Q_full, R_full = QR_con_HH(A, tol)
    
    # dimensiones originales
    m, n = A.shape
    
    # hacemos el recorte
    # Q: Todas las filas, solo n columnas
    Q_red = Q_full[:, :n]
    
    # R: Solo n filas, todas las columnas (que son n)
    R_red = R_full[:n, :]
    
    return Q_red, R_red

def QR_GS_reducida(A, tol=1e-12):
    """
    Calcula QR con Gram-Schmidt.
    Como GS suele trabajar columna a columna, generalmente ya da la reducida,
    pero aplicamos el slice por seguridad para garantizar dimensiones n x n.
    """
    Q_full, R_full = QR_con_GS(A, tol)
    m, n = A.shape
    
    return Q_full[:, :n], R_full[:n, :]

def calculaQR_reducida(A, metodo='RH', tol=1e-12):
    """
    despachador para versiones reducidas.
    """
    if metodo == 'GS':
        return QR_GS_reducida(A, tol)
    elif metodo == 'RH':
        return QR_HH_reducida(A, tol)
    else:
        return None  
    
    
#%%
# Tests L05-QR:

import numpy as np

# --- Matrices de prueba ---
A2 = np.array([[1., 2.],
               [3., 4.]])

A3 = np.array([[1., 0., 1.],
               [0., 1., 1.],
               [1., 1., 0.]])

A4 = np.array([[2., 0., 1., 3.],
               [0., 1., 4., 1.],
               [1., 0., 2., 0.],
               [3., 1., 0., 2.]])

# --- Funciones auxiliares para los tests ---
def check_QR(Q,R,A,tol=1e-10):
    # Comprueba ortogonalidad y reconstrucción
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
    assert np.allclose(Q @ R, A, atol=tol)

# --- TESTS PARA QR_by_GS2 ---
Q2,R2 = QR_con_GS(A2)
check_QR(Q2,R2,A2)

Q3,R3 = QR_con_GS(A3)
check_QR(Q3,R3,A3)

Q4,R4 = QR_con_GS(A4)
check_QR(Q4,R4,A4)

# --- TESTS PARA QR_by_HH ---
Q2h,R2h = QR_con_GS(A2)
check_QR(Q2h,R2h,A2)

Q3h,R3h = QR_con_HH(A3)
check_QR(Q3h,R3h,A3)

Q4h,R4h = QR_con_HH(A4)
check_QR(Q4h,R4h,A4)

# --- TESTS PARA calculaQR ---
Q2c,R2c = calculaQR(A2,metodo='RH')
check_QR(Q2c,R2c,A2)

Q3c,R3c = calculaQR(A3,metodo='GS')
check_QR(Q3c,R3c,A3)

Q4c,R4c = calculaQR(A4,metodo='RH')
check_QR(Q4c,R4c,A4)

#%%
# Laboratorio 6

def multiplicacion_mat_vect(A, v):
    """
    Multiplica una matriz A (m x n) por un vector v (n).
    Retorna un vector de dimensión m.
    """
    A = np.array(A, dtype=float)
    v = np.array(v, dtype=float)

    m, n = A.shape
    if len(v) != n:
        raise ValueError("Dimensiones incompatibles")

    resultado = np.zeros(m)

    for i in range(m):
        resultado[i] = producto_punto(A[i, :], v)

    return resultado

def metpot2k(A, tol=1e-15, K=1000):
    """
    A una matriz simétrica de nxn 
    tol la tolerancia en la diferencia entre un paso y el siguiente de la
    estimación del autovector
    K el numero máximo de iteraciones a realizarse
    retorna vector v, autovalor lamda y numero de iteraciones k
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]

    # 1. Random start
    v = np.random.rand(n)
    
    # 2. Normalizar
    nrm = norma(v, 2)
    v_bar = v / nrm
    
    e = 0.0 
    k = 0

    # 3. Criterio de parada por ANGULO (e), no por lambda
    while abs(e - 1) > tol and k < K:
        v = v_bar.copy()
        
        # Multiplicar
        Av = np.zeros(n)
        for i in range(n):
            Av[i] = np.sum(A[i,:] * v)
        
        # Normalizar
        norm_Av = norma(Av, 2)
        if norm_Av < 1e-15:
            return v, 0.0, k+1
            
        v_bar = Av / norm_Av
        
        # Producto punto (coseno del ángulo)
        e = np.sum(v_bar * v)
        
        # Corrección de signo (si apuntan opuesto, es el mismo autovector)
        if e < 0:
            v_bar = -1.0 * v_bar
            e = -e 
            
        k = k + 1

    # 4. Rayleigh final
    Av_final = np.zeros(n)
    for i in range(n):
        Av_final[i] = np.sum(A[i,:] * v_bar)
        
    lam = np.sum(v_bar * Av_final)
    
    return v_bar, lam, k

def diagRH(A, tol=1e-15, K=1000):
    
    """
    Implementa la diagonalización espectral de matrices simétricas mediante 
    Deflación por Householder recursiva.

    Estrategia del algoritmo:
    1. Reducción: Utiliza el Método de la Potencia para hallar el autovalor y 
       autovector dominante.
    2. Deflación: Construye un reflector de Householder (H) para transformar la 
       matriz, aislando el autovalor en la diagonal y anulando el resto de 
       la primera fila/columna.
    3. Recursión: Aplica el mismo proceso a la submatriz restante (n-1 x n-1) 
       hasta llegar al caso base (1x1).
    4. Reconstrucción: Acumula las transformaciones ortogonales (H) hacia atrás 
       para obtener la matriz final de autovectores S.
    """
    
    """
    A una matriz simétrica de nxn 
    tol la tolerancia en la diferencia entre un paso y el siguiente de la
    estimación del autovector
    K el numero máximo de iteraciones a realizarse
    retorna matriz de autovectores S, matriz de autovalores D, tal que A = SDS^t
    si la matriz A no es simetrica, retorna None
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    """
    # verifico simetria
    # Recorremos solo el triángulo superior sin diagonal
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > tol:
                return None # No es simétrica
    """
    # CASO BASE
    # si la matriz es de 1x1, ya está diagonalizada.
    if n == 1:
        S = np.array([[1.0]])
        D = np.array([[A[0,0]]])
        return S, D

    # PASO 1: Buscar autovalor/vector dominante con metodo de la potencia
    v1, lam1, _ = metpot2k(A, tol, K)
    
    # PASO 2: Construir el Reflector Householder 
    # Queremos rotar v1 para que se alinee con e1 = [1, 0, ..., 0]
    # u = v1 - ||v1|| * e1 (con cuidado de signo)
    
    e1 = np.zeros(n); e1[0] = 1.0
    
    # truco del signo para estabilidad (mismo que en QR)
    signo = 1.0 if v1[0] >= 0 else -1.0
    norm_v1 = norma(v1, 2) # Debería ser 1, pero por las dudas
    
    # u = v1 + sign * ||v1|| * e1 (o la resta del algoritmo, ajustado por signo)
    # El algoritmo dice (e1 - v1), nosotros usamos la versión estable:
    u = v1.copy()
    u[0] += signo * norm_v1 
    
    norm_u = norma(u, 2)
    if norm_u < 1e-15:
        # Si v1 ya es e1, no hay que rotar nada. H = Identidad.
        # Esto pasa si la matriz ya tenía ceros.
        u = np.zeros(n) 
    else:
        u = u / norm_u # Normalizamos u

    # PASO 3: Deflación H A H^t 
    # Calculamos B = H * A * H^t
    # H = I - 2uu^t
    # No multiplicamos matrices. Aplicamos el reflector a filas y columnas.
    
    # Copia de trabajo
    B = A.copy()
    
    # Si u no es nulo, aplicamos la transformación
    if norm_u >= 1e-15:
        # aplico H por izquierda (a las columnas): B = B - 2u(u^t B)
        # B_col_j = B_col_j - 2 * (u . B_col_j) * u
        # Lo hacemos vectorizado con slicing
        for j in range(n):
            prod = np.sum(u * B[:, j]) # u^t * col
            B[:, j] -= 2 * prod * u
            
        # aplico H por derecha (a las filas): B = B - 2(B u)u^t
        # B_fila_i = B_fila_i - 2 * (fila_i . u) * u
        for i in range(n):
            prod = np.sum(B[i, :] * u)
            B[i, :] -= 2 * prod * u

    # PASO 4: Recursión
    # tomo la submatriz (desde 1 en adelante)
    A_tilde = B[1:, 1:]
    
    # llamada recursiva:
    S_tilde, D_tilde = diagRH(A_tilde, tol, K)
    
    # PASO 5: Armar el resultado
    
    # D: Ponemos lambda1 arriba a la izquierda y D_tilde abajo
    D = np.zeros((n, n))
    D[0, 0] = lam1
    D[1:, 1:] = D_tilde
    
    # S: Construimos la matriz mixta y aplicamos H
    # S_temp = [ 1   0     ]
    #          [ 0   S_tilde]
    S_temp = np.eye(n)
    S_temp[1:, 1:] = S_tilde
    
    # S = H * S_temp
    # Aplicamos H a las columnas de S_temp
    S = S_temp.copy()
    if norm_u >= 1e-15:
        for j in range(n):
            # prod = u . columna_j
            prod = np.sum(u * S[:, j])
            S[:, j] -= 2 * prod * u
            
    return S, D

#%%
# Test L06-metpot2k, Aval

# Tests metpot2k

S = np.vstack([
    np.array([2,1,0])/np.sqrt(5),
    np.array([-1,2,5])/np.sqrt(30),
    np.array([1,-2,1])/np.sqrt(6)
              ]).T

# Pedimos que pase el 95% de los casos
exitos = 0
for i in range(100):
    D = np.diag(np.random.random(3)+1)*100
    A = S@D@S.T
    v,l,_ = metpot2k(A,1e-15,1e5)
    if np.abs(l - np.max(D))< 1e-8:
        exitos += 1
assert exitos > 95


#Test con HH
exitos = 0
for i in range(100):
    v = np.random.rand(9)
    #v = np.abs(v)
    #v = (-1) * v
    ixv = np.argsort(-np.abs(v))
    D = np.diag(v[ixv])
    I = np.eye(9)
    H = I - 2*np.outer(v.T, v)/(np.linalg.norm(v)**2)   #matriz de HouseHolder

    A = H@D@H.T
    v,l,_ = metpot2k(A, 1e-15, 1e5)
    #max_eigen = abs(D[0][0])
    if abs(l - D[0,0]) < 1e-8:         
        exitos +=1
assert exitos > 95



# Tests diagRH
D = np.diag([1,0.5,0.25])
S = np.vstack([
    np.array([1,-1,1])/np.sqrt(3),
    np.array([1,1,0])/np.sqrt(2),
    np.array([1,-1,-2])/np.sqrt(6)
              ]).T

A = S@D@S.T
SRH,DRH = diagRH(A,tol=1e-15,K=1e5)
assert np.allclose(D,DRH)
assert np.allclose(np.abs(S.T@SRH),np.eye(A.shape[0]),atol=1e-7)



# Pedimos que pase el 95% de los casos
exitos = 0
for i in range(100):
    A = np.random.random((5,5))
    A = 0.5*(A+A.T)
    S,D = diagRH(A,tol=1e-15,K=1e5)
    ARH = S@D@S.T
    e = normaExacta(ARH-A)[1] # Tomamos el segundo elemento (Norma Infinito)
    if e < 1e-5: 
        exitos += 1
assert exitos >= 95

#%%
# Laboratorio 7
def transiciones_al_azar_continuas(n):
    """
    n cantidad de filas (columnas) de la matriz de transicion
    retorna matriz T de n x n normalizada por columnas, y con entradas al azar en el intervalo
    
    Genera una matriz de n x n donde todos con todos están conectados con probabilidades aleatorias.
    Crea una matriz llena de números al azar en el intervalo [0, 1] y después normaliza cada columna
    para que sume 1.
    Representa un sistema donde es posible ir de cualquier estado a cualquier otro.
    """
    # generar n^2 números aleatorios entre [0, 1]
    m = np.random.random((n, n))

    #normalizo por columna
    for i in range(n):
        #tomo la columna n
        col= m[:, i]
        nor = norma(col, 1) #saco la norma de la col para luego normalizar
        if (nor > 0):
            for j in range(n):
                #normalizo la columna
                m[j,i] = m[j,i] / nor

    return m

def transiciones_al_azar_uniformes(n,thres):
    """
    n cantidad de filas (columnas) de la matriz de transicion
    thres probabilidad de que una entrada sea distinta de cero
    retorna matriz T de nxn normalizada por columnas. 
    El elemento i,j es distinto de cero si el numero generado al azar para i,j
    es menor o igual a thres
    Todos los elementod de la columna j son iguales ( a 1 sobre el numero de elementos 
    distintos de cero)
    
    Genera una matriz donde las conexiones son "a todo o nada" (uniformes). O estás conectado o no lo estás.
    Tira números al azar. Si el número es menor a thres (umbral), pone un 1 (hay conexión); si no, pone un 0.
    """
    # Generar n^2 números aleatorios entre [0, 1]
    m = np.random.random((n, n))

    #pongo caso base si n = 1
    if (n==1 and 1 >= thres):
        m[0] = 1
        return m

    #normalizo por columna
    for i in range(n):
        for j in range(n):
                #comparo con la probabilidad
            if (m[j,i] > thres):
                m[j,i] = 0
            else:
                m[j,i] = 1
    for j in range(n):
        suma_columna = 0
        for i in range(n):
            suma_columna += m[i, j]

        if suma_columna == 0:
            # Si toda la columna es cero, elijo una fila al azar y la pongo en 1
            fila_azar = int(np.random.random() * n)
            m[fila_azar, j] = 1

    #normalizo por columna
    for j in range(n):
            #tomo la columna n
        col= m[:, j]
        nor = norma(col, 1) #saco la norma de la col para luego normalizar
        if (nor > 0):
            for i in range(n):
                m[i,j] = m[i,j]/nor

    return m

def nucleo(A, tol=1e-15):
    """
    Calcula el núcleo de A diagonalizando A.T @ A con diagRH.
    Retorna los vectores del núcleo (columnas).
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    
    # construimos M = A.T * A 
    # M será de nxn.
    M = np.zeros((n, n), dtype=float)

    # Optimizacion: M es simétrica. Calculamos solo triángulo superior.
    for i in range(n):
        for j in range(i, n):
            # El elemento M[i,j] es el producto punto entre:
            # - La fila i de A.T (que es la COLUMNA i de A)
            # - La columna j de A
            
            # Usamos slicing y np.sum para velocidad vectorizada
            val = np.sum(A[:, i] * A[:, j])
            
            M[i, j] = val
            
            # Espejamos al triángulo inferior
            if i != j:
                M[j, i] = val

    # diiagonalizamos M (que sabemos que es simétrica)
    # Traé tu función diagRH del labo anterior
    S, D = diagRH(M, tol=tol) 
    
    # autovectores con autovalor ~ 0
    indices_nucleo = []
    for k in range(n):
        # D es matriz diagonal, miramos D[k,k]
        if abs(D[k, k]) < tol:
            indices_nucleo.append(k)
            
    #Extraemos esos vectores de S
    if len(indices_nucleo) == 0:
        # Retornamos matriz vacía con n filas y 0 columnas
        return np.zeros((n, 0)) 
        
    # Devolvemos las columnas correspondientes
    V_nucleo = S[:, indices_nucleo]
    
    return V_nucleo

def crea_rala(listado, m_filas, n_columnas, tol=1e-15):
    """
    Comprime una matriz. En lugar de guardar un cuadro gigante lleno de ceros, 
    guarda un diccionario (mapa) solo con los valores que importan.
    
    lógica: Recibe listas de índices y valores. Si el valor es mayor a la tolerancia
    (no es cero), lo guarda en el diccionario: {(fila, columna): valor}. 
    Todo lo que no está en el diccionario, se asume que es cero.
    """
    
    if not listado: # chequear lista vacia
        return {}, (m_filas, n_columnas)

    filas = listado[0]
    col = listado[1]
    elementos = listado[2]
    datos = {}

    if len(filas) != len(elementos):
        raise ValueError("Listas de distinta longitud")

    for k in range(len(elementos)):
        # Corregido el parentesis del abs
        if abs(elementos[k]) >= tol:
            datos[(filas[k], col[k])] = elementos[k]

    return datos, (m_filas, n_columnas)

def multiplica_rala_vector(A, v):
    
    """
    Calcula el producto w = Av de forma eficiente.
    lógica: En vez de hacer el producto matricial clásico (fila x columna),
    recorre solo los elementos del diccionario.
    La Optimización: Solo hace cuentas cuando sabe que el elemento de la matriz 
    no es cero (wi = wi + Aij*vj). 
    """
    diccionario, (n, m) = A
    
    if len(v) != m:
        raise ValueError("Dimensiones incompatibles")
        
    # El resultado w tiene tamaño n (filas de A)
    w = np.zeros(n) 
    
    # Iteramos SOLO sobre los elementos no nulos
    # Esto es O(k) donde k es la cantidad de elementos, mucho menor que O(n*m)
    for (i, j), valor in diccionario.items():
        # w[i] += A_ij * v[j]
        w[i] += valor * v[j]
        
    return w

#%% 
# Test Laboratorio 7
def es_markov(T,tol=1e-6):
    """
    T una matriz cuadrada.
    tol la tolerancia para asumir que una suma es igual a 1.
    Retorna True si T es una matriz de transición de Markov (entradas no negativas y columnas que suman 1 dentro de la tolerancia), False en caso contrario.
    """
    n = T.shape[0]
    for i in range(n):
        for j in range(n):
            if T[i,j]<0:
                return False
    for j in range(n):
        suma_columna = sum(T[:,j])
        if np.abs(suma_columna - 1) > tol:
            return False
    return True

def es_markov_uniforme(T,thres=1e-6):
    """
    T una matriz cuadrada.
    thres la tolerancia para asumir que una entrada es igual a cero.
    Retorna True si T es una matriz de transición de Markov uniforme (entradas iguales a cero o iguales entre si en cada columna, y columnas que suman 1 dentro de la tolerancia), False en caso contrario.
    """
    if not es_markov(T,thres):
        return False
    # cada columna debe tener entradas iguales entre si o iguales a cero
    m = T.shape[1]
    for j in range(m):
        non_zero = T[:,j][T[:,j] > thres]
        # all close
        close = all(np.abs(non_zero - non_zero[0]) < thres)
        if not close:
            return False
    return True


def esNucleo(A,S,tol=1e-5):
    """
    A una matriz m x n
    S una matriz n x k
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Retorna True si las columnas de S estan en el nucleo de A (es decir, A*S = 0. Esto no chequea si es todo el nucleo
    """
    for col in S.T:
        res = A @ col
        if not np.allclose(res,np.zeros(A.shape[0]), atol=tol):
            return False
    return True

## TESTS
# transiciones_al_azar_continuas
# transiciones_al_azar_uniformes
for i in range(1,100):
    T = transiciones_al_azar_continuas(i)
    assert es_markov(T), f"transiciones_al_azar_continuas fallo para n={i}"
    
    T = transiciones_al_azar_uniformes(i,0.3)
    assert es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"
    # Si no atajan casos borde, pueden fallar estos tests. Recuerden que suma de columnas DEBE ser 1, no valen columnas nulas.
    T = transiciones_al_azar_uniformes(i,0.01)
    assert es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"
    T = transiciones_al_azar_uniformes(i,0.01)
    assert es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"
    
# nucleo
A = np.eye(3)
S = nucleo(A)
assert S.shape[1]==0, "nucleo fallo para matriz identidad"
A[1,1] = 0
S = nucleo(A)
msg = "nucleo fallo para matriz con un cero en diagonal"
assert esNucleo(A,S), msg
assert S.shape==(3,1), msg
assert abs(S[2,0])<1e-2, msg
assert abs(S[0,0])<1e-2, msg

v = np.random.random(5)
v = v / np.linalg.norm(v)
H = np.eye(5) - np.outer(v, v)  # proyección ortogonal
S = nucleo(H)
msg = "nucleo fallo para matriz de proyeccion ortogonal"
assert S.shape==(5,1), msg
v_gen = S[:,0]
v_gen = v_gen / np.linalg.norm(v_gen)
assert np.allclose(v, v_gen) or np.allclose(v, -v_gen), msg

# crea rala
listado = [[0,17],[3,4],[0.5,0.25]]
A_rala_dict, dims = crea_rala(listado,32,89)
assert dims == (32,89), "crea_rala fallo en dimensiones"
assert A_rala_dict[(0,3)] == 0.5, "crea_rala fallo"
assert A_rala_dict[(17,4)] == 0.25, "crea_rala fallo"
assert len(A_rala_dict) == 2, "crea_rala fallo en cantidad de elementos"

listado = [[32,16,5],[3,4,7],[7,0.5,0.25]]
A_rala_dict, dims = crea_rala(listado,50,50)
assert dims == (50,50), "crea_rala fallo en dimensiones con tol"
assert A_rala_dict.get((32,3)) == 7
assert A_rala_dict[(16,4)] == 0.5
assert A_rala_dict[(5,7)] == 0.25

listado = [[1,2,3],[4,5,6],[1e-20,0.5,0.25]]
A_rala_dict, dims = crea_rala(listado,10,10)
assert dims == (10,10), "crea_rala fallo en dimensiones con tol"
assert (1,4) not in A_rala_dict
assert A_rala_dict[(2,5)] == 0.5
assert A_rala_dict[(3,6)] == 0.25
assert len(A_rala_dict) == 2

# caso borde: lista vacia. Esto es una matriz de 0s
listado = []
A_rala_dict, dims = crea_rala(listado,10,10)
assert dims == (10,10), "crea_rala fallo en dimensiones con lista vacia"
assert len(A_rala_dict) == 0, "crea_rala fallo en cantidad de elementos con lista vacia"

# multiplica rala vector
listado = [[0,1,2],[0,1,2],[1,2,3]]
A_rala = crea_rala(listado,3,3)
v = np.random.random(3)
v = v / np.linalg.norm(v)
res = multiplica_rala_vector(A_rala,v)
A = np.array([[1,0,0],[0,2,0],[0,0,3]])
res_esperado = A @ v
assert np.allclose(res,res_esperado), "multiplica_rala_vector fallo"

A = np.random.random((5,5))
A = A * (A > 0.5) 
listado = [[],[],[]]
for i in range(5):
    for j in range(5):
        listado[0].append(i)
        listado[1].append(j)
        listado[2].append(A[i,j])
        
A_rala = crea_rala(listado,5,5)
v = np.random.random(5)
assert np.allclose(multiplica_rala_vector(A_rala,v), A @ v)

#%%
# Laboratorio 8

def transpuesta(A):
    rows, cols = A.shape
    AT = np.zeros((cols, rows))
    for i in range(rows):
        for j in range(cols):
            AT[j, i] = A[i, j]
    return AT

def sort_desc(vals):
    """
    Retorna los índices que ordenarían el array 'vals' de MAYOR a MENOR.
    Reemplaza a np.argsort(vals)[::-1]
    """
    n = len(vals)
    # Creamos la lista de índices [0, 1, 2, ..., n-1]
    indices = list(range(n))
    
    # Bubble sort aplicado a los índices basándose en los valores
    for i in range(n):
        for j in range(0, n-i-1):
            # Comparamos los valores correspondientes a los índices actuales
            # Si el valor de la izquierda es MENOR que el de la derecha, intercambiamos
            # (porque queremos orden DESCENDENTE: los grandes al principio)
            if vals[indices[j]] < vals[indices[j+1]]:
                # Intercambiamos los índices
                indices[j], indices[j+1] = indices[j+1], indices[j]
                
    return indices

def svd_reducida(A, k="max", tol=1e-15):
    """
    Calcula la SVD reducida: A = U * Sig * V^t
    """
    A = np.array(A, dtype=float)
    m, n = A.shape

    # Caso 1: MATRIZ ANCHA (m < n)
    if m < n:
        # Llamada recursiva con la transpuesta
        At = transpuesta(A)
        # Obtenemos (V, Sig, U) de la transpuesta
        V_hat, hatSig, U_hat = svd_reducida(At, k, tol)
        # Retornamos permutado
        return U_hat, hatSig, V_hat

    # Caso 2: MATRIZ ALTA O CUADRADA (m >= n)
    # Construyo la matriz simetrica M = A^t * A
    At = transpuesta(A)
    M = multiplicacion_mat(At, A)

    # Diagonalizamos M (obtenemos V y los valores singulares al cuadrado)
    V_full, D = diagRH(M, tol=tol)

    # Procesamos Valores Singulares
    sigmas_cuadrado = []
    for i in range(n):
        sigmas_cuadrado.append(D[i, i])
    
    # ahora obtenemos los índices ordenados de mayor a menor manualmente
    indices = sort_desc(sigmas_cuadrado)
    
    sigmas_validos = []
    indices_validos = []
    
    for idx in indices:
        val = sigmas_cuadrado[idx]
        # Filtramos los muy chicos o negativos (ruido numérico)
        if val > tol:
            sigmas_validos.append(np.sqrt(val))
            indices_validos.append(idx)
    
    # truncamiento por k
    if k == "max":
        r = len(sigmas_validos)
    else:
        # Si k es un número, tomamos el mínimo entre k y los que encontramos
        r = min(int(k), len(sigmas_validos))
        
    hatSig = np.array(sigmas_validos[:r])
    
    # construimos las matrices recortadas
    # hatV: Reordenamos las columnas de V_full según los índices ganadores
    hatV = np.zeros((n, r))
    for i in range(r):
        orig_idx = indices_validos[i]
        hatV[:, i] = V_full[:, orig_idx]

    # hatU: se calcula como A * v_i / sigma_i
    hatU = np.zeros((m, r))
    
    for i in range(r):
        sigma = hatSig[i]
        v_col = hatV[:, i]
        
        # u_i = (A * v_i) / sigma
        # Multiplicación Matriz-Vector manual
        Av = np.zeros(m)
        for row in range(m):
            Av[row] = np.sum(A[row, :] * v_col)
            
        hatU[:, i] = Av / sigma

    return hatU, hatSig, hatV

#%%
# Tests L08
# Matrices al azar
def genera_matriz_para_test(m,n=2,tam_nucleo=0):
    if tam_nucleo == 0:
        A = np.random.random((m,n))
    else:
        A = np.random.random((m,tam_nucleo))
        A = np.hstack([A,A])
    return(A)

def test_svd_reducida_mn(A,tol=1e-15):
    m,n = A.shape
    hU,hS,hV = svd_reducida(A,tol=tol)
    nU,nS,nVT = np.linalg.svd(A)
    r = len(hS)+1
    assert np.all(np.abs(np.abs(np.diag(hU.T @ nU))-1)<10**r*tol), 'Revisar calculo de hat U en ' + str((m,n))
    assert np.all(np.abs(np.abs(np.diag(nVT @ hV))-1)<10**r*tol), 'Revisar calculo de hat V en ' + str((m,n))
    assert len(hS) == len(nS[np.abs(nS)>tol]), 'Hay cantidades distintas de valores singulares en ' + str((m,n))
    assert np.all(np.abs(hS-nS[np.abs(nS)>tol])<10**r*tol), 'Hay diferencias en los valores singulares en ' + str((m,n))

for m in [2,5,10,20]:
    for n in [2,5,10,20]:
        for _ in range(10):
            A = genera_matriz_para_test(m,n)
            test_svd_reducida_mn(A)


# Matrices con nucleo

m = 12
for tam_nucleo in [2,4,6]:
    for _ in range(10):
        A = genera_matriz_para_test(m,tam_nucleo=tam_nucleo)
        test_svd_reducida_mn(A)

# Tamaños de las reducidas
A = np.random.random((8,6))
for k in [1,3,5]:
    hU,hS,hV = svd_reducida(A,k=k)
    assert hU.shape[0] == A.shape[0], 'Dimensiones de hU incorrectas (caso a)'
    assert hV.shape[0] == A.shape[1], 'Dimensiones de hV incorrectas(caso a)'
    assert hU.shape[1] == k, 'Dimensiones de hU incorrectas (caso a)'
    assert hV.shape[1] == k, 'Dimensiones de hV incorrectas(caso a)'
    assert len(hS) == k, 'Tamaño de hS incorrecto'


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###############################################################################
# COMIENZO DE LOS EJERCICIOS DEL ENUNCIADO DEL TP
#Ejercicio 1 - cargar dataset

def _ensure_2d_columns(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    elif arr.ndim == 2:
        return arr
    else:
        raise ValueError(f"Array con más de 2 dimensiones no esperado: shape={arr.shape}")

def cargarDataset(carpeta):
    """
    Carga embeddings de gatos y perros (train y val) desde una carpeta base.
    Estructura esperada:
        carpeta/+
         ├── train/
         │   ├── cats/
         │   └── dogs/
         └── val/
             ├── cats/
             └── dogs/
    En caso de estructura distinta, ajustar en USO mas abajo
    Devuelve: Xt, Yt, Xv, Yv
    """

    # rutas correctas según tu estructura real
    train_cats = os.path.join(carpeta, "train", "cats", "efficientnet_b3_embeddings.npy")
    train_dogs = os.path.join(carpeta, "train", "dogs", "efficientnet_b3_embeddings.npy")
    val_cats   = os.path.join(carpeta, "val", "cats", "efficientnet_b3_embeddings.npy")
    val_dogs   = os.path.join(carpeta, "val", "dogs", "efficientnet_b3_embeddings.npy")

    # cargar embeddings
    Xt_c = _ensure_2d_columns(np.load(train_cats))
    Xt_d = _ensure_2d_columns(np.load(train_dogs))
    Xv_c = _ensure_2d_columns(np.load(val_cats))
    Xv_d = _ensure_2d_columns(np.load(val_dogs))

    # concatenar embeddings (gatos primero, perros después)
    Xt = np.concatenate((Xt_c, Xt_d), axis=1)
    Xv = np.concatenate((Xv_c, Xv_d), axis=1)

    # crear etiquetas
    n_cat_train = Xt_c.shape[1]
    n_dog_train = Xt_d.shape[1]
    Yt = np.concatenate((
        np.tile([[1], [0]], (1, n_cat_train)),  # gatos
        np.tile([[0], [1]], (1, n_dog_train))   # perros
    ), axis=1)

    n_cat_val = Xv_c.shape[1]
    n_dog_val = Xv_d.shape[1]
    Yv = np.concatenate((
        np.tile([[1], [0]], (1, n_cat_val)),
        np.tile([[0], [1]], (1, n_dog_val))
    ), axis=1)

    return Xt, Yt, Xv, Yv

# === USO ===
if __name__ == "__main__":
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    carpeta = os.path.join(ruta_actual, "template-alumnos", "cats_and_dogs")

    Xt, Yt, Xv, Yv = cargarDataset(carpeta)
    print("Xt:", Xt.shape)
    print("Yt:", Yt.shape)
    print("Xv:", Xv.shape)
    print("Yv:", Yv.shape)
    print("Primeras columnas de Yt:\n", Yt[:, :10])

#%% COMIENZO DE LOS ALGORITMOS
#Ejercicio 2, Algoritmo 1. Cholesky con rango completo. 

def cholesky(A):
    n = len(A)
    
    # verificar SIMETRÍA 
    # Recorremos solo el triángulo superior sin diagonal
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > 1e-8: # si son casi cero
                    raise ValueError("La matriz no es definida positiva, falla Cholesky.")
                    
    #puedo optimizar esta parte del código con algunas funciones de numpy:
    #if np.max(np.abs(A - A.T)) > 1e-8:
    #    raise ValueError("La matriz no es simétrica. Cholesky requiere simetría.")
            
    # Inicializar L con ceros
    L = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):  # solo en la parte triangular inferior
            suma = sum(L[i][k] * L[j][k] for k in range(j))

            if i == j:
                # Elementos en la diagonal
                val = A[i][i] - suma
                if val <= 0:
                    raise ValueError("La matriz no es definida positiva, falla Cholesky.")
                L[i][j] = val**0.5
            else:
                # Elementos fuera de la diagonal
                L[i][j] = (A[i][j] - suma) / L[j][j]

    return np.array(L, dtype=float)

#%%

# Función para calcular el rango de una matriz y poder clasificar en a), b) o c) segun corresponda.
def calcular_rango(A):
    M = A.astype(float).copy()   # copia para no modificar A
    n, m = M.shape
    fila = 0
    eps = 1e-12

    for col in range(m):
        # Buscar pivote en la columna col
        piv = None
        for i in range(fila, n):
            if abs(M[i, col]) > eps:
                piv = i
                break

        # Si no hay pivote, columna dependiente
        if piv is None:
            continue

        # Intercambiar filas: pivote → fila actual
        if piv != fila:
            M[[fila, piv]] = M[[piv, fila]]

        # Normalizar fila pivote
        piv_val = M[fila, col]
        M[fila, col:] = M[fila, col:] / piv_val

        # Eliminar debajo del pivote
        for i in range(fila + 1, n):
            factor = M[i, col]
            M[i, col:] -= factor * M[fila, col:]

        fila += 1

        # Terminamos si usamos todas las filas
        if fila == n:
            break

    return fila

#%%
# Implementa los tres casos del Algoritmo 1 (a, b, c).

# Recibe:
# X : matriz n × p
# L : matriz Cholesky
# Y : matriz n × c

def pinvEcuacionesNormales(X, L, Y):

    # Dimensiones de X:
    n = len(X)        # filas
    p = len(X[0])     # columnas

    # Rango de X: 
    X_np = np.array(X, dtype=float)
    rango = calcular_rango(X_np)

    # CASO (a y c): rango(X) = p =< n
    # Cholesky de X^t X
    # CAMBIO:
    # resolver W = Y (X^t X)^(-1) X^t para una matriz cuadrada e invertible
    # desgloso el -1: W = Y (X^(-1) (X^t)^(-1)) X^t y (X^t)^(-1)*X^t = I -->> entonces me queda: W = Y * X^(-1) -> caso c)
    # Matemáticamente se simplifica a W = Y X^(-1), que es exactamente lo que pide el algoritmo para el caso cuadrado.
    # Y computacionalmente, uso Cholesky de X^t X, que siempre es simétrica y definida positiva, evitando el error de
    # intentar Cholesky sobre una X asimétrica.
    if rango == p and n >= p:
        print("Caso (a) detectado: rango(X)=p<n. Usando (X^T X).")

        XT = transpuesta(X)

        # Primer sistema: L Z = X^T
        Z_cols = []
        for j in range(len(XT[0])):             
            b = [XT[i][j] for i in range(len(XT))]
            z = res_tri(L, b, inferior=True)
            Z_cols.append(z)
        Z = transpuesta(Z_cols)

        # Segundo sistema: L^T U = Z
        LT = transpuesta(L)
        U_cols = []
        for j in range(len(Z[0])):              
            b = [Z[i][j] for i in range(len(Z))]
            u = res_tri(LT, b, inferior=False)
            U_cols.append(u)
        U = transpuesta(U_cols)

        #  W = YU
        W = multiplicacion_mat(Y, U)
        return np.array(W, dtype=float)


    # CASO (b): rango(X) = n < p
    #     Cholesky de X X^T
    elif rango == n and n < p:
        print("Caso (b) detectado: rango(X)=n<p. Usando (X X^T).")

        # Primer sistema: L Z = X
        Z_cols = []
        for j in range(len(X[0])):               
            b = [X[i][j] for i in range(len(X))]
            z = res_tri(L, b, inferior=True)
            Z_cols.append(z)
        Z = transpuesta(Z_cols)

        # Segundo sistema: L^T V^T = Z
        LT = transpuesta(L)
        VT_cols = []
        for j in range(len(Z[0])):               
            b = [Z[i][j] for i in range(len(Z))]
            v_t = res_tri(LT, b, inferior=False)
            VT_cols.append(v_t)
        VT = transpuesta(VT_cols)

        # V
        V = transpuesta(VT)

        # W = Y V
        W = multiplicacion_mat(Y, V)
        return np.array(W, dtype=float)

    # Caso inválido
    else:
        raise ValueError(
            f"La matriz X no cumple las condiciones para los casos (a), (b) o (c). "
            f"rango(X)={rango}, n={n}, p={p}"
        )

#%%

# Entreno mi modelo con la L obtenida e Y de entrenamiento, para obtener W
print("\nProbando Ecuaciones Normales...")
# Hago cholesky con mi X de entrenamiento
L = cholesky(multiplicacion_mat(Xt, transpuesta(Xt)))
w_EN = pinvEcuacionesNormales(Xt, L, Yt)
print("W (Ecuaciones Normales):", w_EN.shape)

#%%
#Ejercicio 3, Algoritmo 2. Descomposición en Valores Singulares
#funcion auxiliar para los valores singulares
def pinvSVD(U, S, V, Y, tol=1e-12):

    n = len(S)        # rango

    # ---- 1) Extraer U1 y V1 (las primeras n columnas) ----
    U1 = [fila[:n] for fila in U]   # cada "row" es una fila de U
    V1 = [fila[:n] for fila in V]   # cada "row" es una fila de V

    # ---- 2) Construir Σ1^{-1} (matriz diagonal) ----
    Sigma_inv = [[0.0]*n for _ in range(n)]
    for i in range(n):
        if abs(S[i]) > tol:
            Sigma_inv[i][i] = 1.0 / S[i]

    # ---- 3) V1 Σ1^{-1} ----
    V1_Sinv = multiplicacion_mat(V1, Sigma_inv)

    # ---- 4) U1^T ----
    U1_T = transpuesta(U1)

    # ---- 5) X⁺ = V1 Σ1^{-1} U1^T ----
    X_pinv = multiplicacion_mat(V1_Sinv, U1_T)

    # ---- 6) W = Y X⁺ ----
    W = multiplicacion_mat(Y, X_pinv)

    return W

# ======== PROBAR SVD ==========
print("\nProbando SVD...")
U, S, V = svd_reducida(Xt)
W_svd = pinvSVD(U, S, V, Yt) 
print("W (SVD):", W_svd.shape)

#%%
# Ejercicio 4,  Algoritmo 3. Descomposicion QR
# a continuación, estas funciones calculan los pesos W resolviendo el problema de mínimos cuadrados
# W = YX^+ usando la pseudoinversa obtenida mediante la descomposición QR.

#Primero se factoriza A= X^t =QR. A partir de esta factorización, la pseudoinversa se expresa como:
# X^+ = Q(R^−t)
# Pero en lugar de invertir matrices, resolvemos los sistemas triangulares:
# 1-  Z = Q^t Y
# 2-  R^t V = Z
#y finalmente: W=V^t

#Este procedimiento evita calcular la inversa explícita, es numéricamente estable y funciona tanto 
#con Householder como con Gram–Schmidt.


#tengo que X+ = V y VR^t = Q
#sé que R es triangular superior, entonces R^t es triangular inferior
#entonces transpongo la ecuación anterior: (VR^t)^t = (Q)^t => RV^t = Q^t
#así, puedo hallar V^t mediante la resolucion Ly = b (con res_tri), y así consigo V.
#Luego tengo que W = YV


def pinvHouseHolder(Q, R, Y):
    
    #Q tiene dimension 2000*1536
    #R tiene dimension 1536 * 1536
    # Resolver R * v_j  = q_j
    V = []
    for qj in Q:
        vj = res_tri(R, qj, inferior=False)  
        V.append(vj)

    V = np.array(V)

    W = multiplicacion_mat(Y,V)
    return W


def pinvGramSchmidt(Q, R, Y):

    V = []
    for qj in Q:
        vj = res_tri(R, qj, inferior=False) 
        V.append(vj)

    V = np.array(V)
    return multiplicacion_mat(Y,V)


# ======== PROBAR HOUSEHOLDER ==========
print("\nProbando Householder...")
Qh, Rh = calculaQR_reducida(Xt.T, metodo='RH')
Wh = pinvHouseHolder(Qh, Rh, Yt)

# ======== PROBAR GRAM–SCHMIDT =========
print("\nProbando Gram-Schmidt...")
Qg, Rg = calculaQR_reducida(Xt.T, metodo='GS')
Wg = pinvGramSchmidt(Qg, Rg, Yt)
print("W (GramSchmidt):", Wg.shape)

#%%
#Ejercicio 5.
def esPseudoInversa(X, pX, tol= 18e-8):
    #tiene que cumplir las condiciones de Moore-Penrose
    #1) XpXX = X
    #2) pXXpX = pX
    #3) pXX_t = XpX
    #4) pXX_t = pXX
    
    #multiplicaciones necesarias
    XpX = multiplicacion_mat(X, pX)
    pXX = multiplicacion_mat(pX, X)
    XpXX = multiplicacion_mat(XpX, X)
    pXXpX = multiplicacion_mat(pXX, pX)

    #transpuestas
    XpX_t = transpuesta(XpX)
    pXX_t = transpuesta(pXX)
    
    # Condiciones de Moore–Penrose
    c1 = matricesIguales(XpXX, X, tol)
    c2 = matricesIguales(pXXpX, pX, tol)
    c3 = matricesIguales(XpX_t, XpX, tol)
    c4 = matricesIguales(pXX_t, pXX, tol)
    
    return c1 and c2 and c3 and c4


#%%
# definimos la función argmax. Toma una matriz (y_pred_prob) y nos devuelve una matriz que: por columna, 
# en la posición del número mayor (con más probabilidad), va un 0, en el de menos, va un 1.
# Esta matriz resultante nos permite compararla con Yv para generar nuestra matriz de confusión luego.

def argmax(M):
    filas, cols = M.shape
    if filas != 2:
        raise ValueError("Esta función solo funciona con matrices 2 × n.")

    # crear matriz de salida llena de ceros
    y_pred = np.array([[0 for _ in range(cols)] for _ in range(2)])


    # recorrer columnas
    for j in range(cols):
        # comparar manualmente los dos valores de la columna j
        if M[0][j] >= M[1][j]:
            y_pred[0][j] = 1
        else:
            y_pred[1][j] = 1

    return y_pred
#%%
#hacemos las multiplicaciones matriciales para obtener los resultados Y_res de 
#cada modelo probados en Xv y entrenados en Xt. y a cada resultado le implementamos 
#arg_max, que nos transforma la probabilidad, a un numero entero entre [0,1]

#Ecuaciones normales
y_res_EN = multiplicacion_mat(w_EN, Xv)
y_pred_EN = argmax(y_res_EN)

#Descomposición en Valores Singulares

y_res_SVD = multiplicacion_mat(W_svd, Xv)
y_pred_SVD = argmax(y_res_SVD)

#Descomposición QR con Gram Schmidt
y_res_QR_GS = multiplicacion_mat(Wg, Xv)
y_pred_QR_GS = argmax(y_res_QR_GS)

#Descomposición QR con HouseHolder
y_res_QR_HH = multiplicacion_mat(Wh, Xv)
y_pred_QR_HH = argmax(y_res_QR_HH)

#%%

#============================================================
# guardamos las variables necesarias para poder ejecutar el código de nuestro jupyter
# notebook adjunto en la entrega, en un archivo .npy

np.save('Xv.npy', Xv)
np.save('Yv.npy', Yv)
np.save('y_pred_EN', y_pred_EN)
np.save('y_pred_SVD', y_pred_SVD)
np.save('y_pred_GS', y_pred_QR_GS)
np.save('y_pred_HH', y_pred_QR_HH)

