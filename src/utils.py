#utils.py
# FUNCIÓN PARA VALIDAR EL FORMATO DE UNA MATRÍCULA
def validar_matricula(texto):

    # Comprobar que la matrícula tiene exactamente 4 caracteres
    # El sistema asume que las matrículas están formadas por 4 dígitos
    if len(texto) != 4:
        return False

    # Comprobar que todos los caracteres son números
    # Si contiene letras u otros símbolos se considera inválida
    if not texto.isdigit():
        return False

    # Si pasa todas las comprobaciones, la matrícula es válida
    return True