#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>
#include <ctype.h>

/* Busca el numero menor en un arreglo */
int menorArreglo(int arreglo[], int n) {
    if (arreglo == NULL || n <= 0) {
        return INT_MAX;
    }
    int menor = arreglo[0];
    for (int i = 1; i < n; i++) {
        if (arreglo[i] < menor) {
            menor = arreglo[i];
        }
    }
    return menor;
}

/* Convierte minutos a segundos */
float minutosASegundos(float minutos) {
    return minutos * 60.0f;
}

/* Revisa si una cadena esta vacia */
bool cadenaVacia(const char texto[]) {
    if (texto == NULL) {
        return true;
    }
    return texto[0] == '\0';
}

/* Calcula el area de un cuadrado */
float areaCuadrado(float lado) {
    if (lado <= 0.0f) {
        return 0.0f;
    }
    return lado * lado;
}

/* Cuenta cuantas letras mayusculas tiene una cadena */
int contarMayusculas(const char texto[]) {
    if (texto == NULL) {
        return -1;
    }
    int contador = 0;
    for (int i = 0; texto[i] != '\0'; i++) {
        if (texto[i] >= 'A' && texto[i] <= 'Z') {
            contador++;
        }
    }
    return contador;
}

/* Suma dos numeros enteros */
int sumaEnteros(int a, int b) {
    return a + b;
}

/* Verifica si un numero entero es positivo */
bool numeroPositivo(int num) {
    return num > 0;
}

/* Convierte horas a minutos */
float horasAMinutos(float horas) {
    return horas * 60.0f;
}

/* Cuenta los numeros pares de un arreglo */
int contarPares(int arreglo[], int n) {
    if (arreglo == NULL || n <= 0) {
        return 0;
    }
    int contador = 0;
    for (int i = 0; i < n; i++) {
        if (arreglo[i] % 2 == 0) {
            contador++;
        }
    }
    return contador;
}

/* Calcula el promedio de tres notas */
float promedioTresNotas(float nota1, float nota2, float nota3) {
    return (nota1 + nota2 + nota3) / 3.0f;
}

/* Convierte grados Celsius a Fahrenheit */
float celsiusAFahrenheit(float celsius) {
    return (celsius * 9.0f / 5.0f) + 32.0f;
}

/* Verifica si una edad es de mayor de edad */
bool mayorDeEdad(int edad) {
    return edad >= 18;
}

/* Cuenta cuantos espacios tiene una cadena */
int contarEspacios(const char texto[]) {
    if (texto == NULL) {
        return -1;
    }
    int contador = 0;
    for (int i = 0; texto[i] != '\0'; i++) {
        if (texto[i] == ' ') {
            contador++;
        }
    }
    return contador;
}

/* Obtiene el numero mayor de tres enteros */
int mayorDeTres(int a, int b, int c) {
    int mayor = a;
    if (b > mayor) {
        mayor = b;
    }
    if (c > mayor) {
        mayor = c;
    }
    return mayor;
}

/* Convierte metros a centimetros */
float metrosACentimetros(float metros) {
    return metros * 100.0f;
}

/* Revisa si un numero esta en un rango cerrado */
bool enRango(int num, int min, int max) {
    return num >= min && num <= max;
}

/* Cuenta los numeros negativos de un arreglo */
int contarNegativos(int arreglo[], int n) {
    if (arreglo == NULL || n <= 0) {
        return 0;
    }
    int contador = 0;
    for (int i = 0; i < n; i++) {
        if (arreglo[i] < 0) {
            contador++;
        }
    }
    return contador;
}

/* Calcula el area de un circulo */
float areaCirculo(float radio) {
    const float pi = 3.1415926f;
    if (radio <= 0.0f) {
        return 0.0f;
    }
    return pi * radio * radio;
}

/* Cambia los espacios por guion bajo */
void espaciosAGuionBajo(char texto[]) {
    if (texto == NULL) {
        return;
    }
    for (int i = 0; texto[i] != '\0'; i++) {
        if (texto[i] == ' ') {
            texto[i] = '_';
        }
    }
}

/* Calcula el precio final con descuento */
float precioConDescuento(float precio, float descuento) {
    if (precio < 0.0f || descuento < 0.0f) {
        return 0.0f;
    }
    return precio - (precio * descuento / 100.0f);
}

/* Calcula el factorial de un entero */
long factorialEntero(int num) {
    if (num < 0) {
        return -1;
    }
    long resultado = 1;
    for (int i = 2; i <= num; i++) {
        resultado *= i;
    }
    return resultado;
}

/* Verifica si un numero entero es par */
bool numeroPar(int num) {
    return num % 2 == 0;
}

/* Calcula la potencia de un entero */
long potenciaEntera(int base, int exponente) {
    if (exponente < 0) {
        return 0;
    }
    long resultado = 1;
    for (int i = 0; i < exponente; i++) {
        resultado *= base;
    }
    return resultado;
}

/* Copia una cadena en otra con limite */
void copiarTextoSeguro(char destino[], const char origen[], int limite) {
    if (destino == NULL || origen == NULL || limite <= 0) {
        return;
    }
    int i = 0;
    while (i < limite - 1 && origen[i] != '\0') {
        destino[i] = origen[i];
        i++;
    }
    destino[i] = '\0';
}

/* Convierte una letra minuscula a mayuscula */
char minAMayus(char letra) {
    if (letra >= 'a' && letra <= 'z') {
        return letra - 32;
    }
    return letra;
}

/* Verifica si un numero entero es multiplo de cinco */
bool multiploDeCinco(int num) {
    return num % 5 == 0;
}

/* Calcula el total con impuesto */
float totalConImpuesto(float precio, float impuesto) {
    if (precio < 0.0f || impuesto < 0.0f) {
        return 0.0f;
    }
    return precio + (precio * impuesto / 100.0f);
}

/* Cuenta los digitos que aparecen en una cadena */
int contarDigitos(const char texto[]) {
    if (texto == NULL) {
        return -1;
    }
    int contador = 0;
    for (int i = 0; texto[i] != '\0'; i++) {
        if (isdigit((unsigned char)texto[i])) {
            contador++;
        }
    }
    return contador;
}

/* Calcula el volumen de una esfera */
float volumenEsfera(float radio) {
    const float pi = 3.1415926f;
    if (radio <= 0.0f) {
        return 0.0f;
    }
    return (4.0f * pi * radio * radio * radio) / 3.0f;
}

/* Limpia una cadena de texto */
void limpiarTexto(char texto[]) {
    if (texto == NULL) {
        return;
    }
    texto[0] = '\0';
}

/* Calcula el residuo sin usar modulo */
int residuoSinModulo(int dividendo, int divisor) {
    if (divisor == 0) {
        return 0;
    }
    int cociente = dividendo / divisor;
    return dividendo - (cociente * divisor);
}

/* Busca el numero mayor en un arreglo */
int mayorArreglo(int arreglo[], int n) {
    if (arreglo == NULL || n <= 0) {
        return INT_MIN;
    }
    int mayor = arreglo[0];
    for (int i = 1; i < n; i++) {
        if (arreglo[i] > mayor) {
            mayor = arreglo[i];
        }
    }
    return mayor;
}

/* Calcula el promedio de un arreglo */
double promedioArreglo(int arreglo[], int n) {
    if (arreglo == NULL || n <= 0) {
        return 0.0;
    }
    long suma = 0;
    for (int i = 0; i < n; i++) {
        suma += arreglo[i];
    }
    return (double)suma / n;
}

/* Revisa si una calificacion esta aprobada */
bool calificacionAprobada(float calif) {
    return calif >= 70.0f;
}

/* Convierte kilometros a metros */
float kmAMetros(float km) {
    return km * 1000.0f;
}

/* Invierte los signos de un arreglo */
void invertirSignos(int arreglo[], int n) {
    if (arreglo == NULL || n <= 0) {
        return;
    }
    for (int i = 0; i < n; i++) {
        arreglo[i] = arreglo[i] * -1;
    }
}

/* Cuenta cuantas veces aparece un caracter */
int contarCaracter(const char texto[], char buscado) {
    if (texto == NULL) {
        return -1;
    }
    int contador = 0;
    for (int i = 0; texto[i] != '\0'; i++) {
        if (texto[i] == buscado) {
            contador++;
        }
    }
    return contador;
}

/* Calcula el doble de un entero */
int dobleEntero(int num) {
    return num * 2;
}

/* Genera un numero aleatorio en un rango */
int randomRango(int min, int max) {
    if (max < min) {
        return min;
    }
    return min + rand() % (max - min + 1);
}

/* Verifica si un numero entero es cero */
bool numeroCero(int num) {
    return num == 0;
}

/* Calcula la distancia entre dos puntos de una recta */
float distanciaRecta(float a, float b) {
    float distancia = b - a;
    if (distancia < 0.0f) {
        distancia = distancia * -1.0f;
    }
    return distancia;
}

/* Calcula el area de un triangulo */
float areaTriangulo(float base, float altura) {
    if (base <= 0.0f || altura <= 0.0f) {
        return 0.0f;
    }
    return (base * altura) / 2.0f;
}

/* Intercambia dos valores enteros */
void intercambiarEnteros(int *a, int *b) {
    if (a == NULL || b == NULL) {
        return;
    }
    int aux = *a;
    *a = *b;
    *b = aux;
}

/* Convierte libras a kilogramos */
float librasAKilos(float libras) {
    return libras * 0.453592f;
}

/* Revisa si una letra es vocal minuscula */
bool vocalMinuscula(char letra) {
    return letra == 'a' || letra == 'e' || letra == 'i' || letra == 'o' || letra == 'u';
}

/* Cuenta los alumnos aprobados en un arreglo */
int contarAprobados(float califs[], int n) {
    if (califs == NULL || n <= 0) {
        return 0;
    }
    int contador = 0;
    for (int i = 0; i < n; i++) {
        if (califs[i] >= 70.0f) {
            contador++;
        }
    }
    return contador;
}

/* Resta dos numeros enteros */
int restaEnteros(int a, int b) {
    return a - b;
}

/* Convierte una cadena a minusculas */
void textoAMinusculas(char texto[]) {
    if (texto == NULL) {
        return;
    }
    for (int i = 0; texto[i] != '\0'; i++) {
        texto[i] = (char)tolower((unsigned char)texto[i]);
    }
}

/* Verifica si un numero entero es impar */
bool numeroImpar(int num) {
    return num % 2 != 0;
}

/* Calcula el perimetro de un cuadrado */
float perimetroCuadrado(float lado) {
    if (lado <= 0.0f) {
        return 0.0f;
    }
    return lado * 4.0f;
}

/* Cuenta los numeros positivos de un arreglo */
int contarPositivos(int arreglo[], int n) {
    if (arreglo == NULL || n <= 0) {
        return 0;
    }
    int contador = 0;
    for (int i = 0; i < n; i++) {
        if (arreglo[i] > 0) {
            contador++;
        }
    }
    return contador;
}

/* Compara si dos cadenas son iguales */
bool textosIguales(const char a[], const char b[]) {
    if (a == NULL || b == NULL) {
        return false;
    }
    return strcmp(a, b) == 0;
}

/* Multiplica dos numeros enteros */
int productoEnteros(int a, int b) {
    return a * b;
}

/* Calcula la raiz cuadrada segura */
float raizSegura(float num) {
    if (num < 0.0f) {
        return 0.0f;
    }
    return sqrtf(num);
}

/* Cuenta los numeros cero de un arreglo */
int contarCeros(int arreglo[], int n) {
    if (arreglo == NULL || n <= 0) {
        return 0;
    }
    int contador = 0;
    for (int i = 0; i < n; i++) {
        if (arreglo[i] == 0) {
            contador++;
        }
    }
    return contador;
}

/* Verifica si un caracter es una letra */
bool esLetra(char c) {
    return isalpha((unsigned char)c) != 0;
}

/* Normaliza una calificacion entre cero y cien */
int normalizarCalificacion(int calif) {
    if (calif < 0) {
        return 0;
    }
    if (calif > 100) {
        return 100;
    }
    return calif;
}

/* Calcula la division segura de dos flotantes */
float divisionSegura(float a, float b) {
    if (b == 0.0f) {
        return 0.0f;
    }
    return a / b;
}

/* Cuenta caracteres de una cadena sin strlen */
int contarCaracteres(const char texto[]) {
    if (texto == NULL) {
        return -1;
    }
    int contador = 0;
    while (texto[contador] != '\0') {
        contador++;
    }
    return contador;
}

/* Convierte litros a mililitros */
float litrosAMililitros(float litros) {
    return litros * 1000.0f;
}

