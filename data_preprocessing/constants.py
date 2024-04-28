import periodictable
# https://periodictable.readthedocs.io/en/latest/guide/using.html
# https://www.geeksforgeeks.org/get-the-details-of-an-element-by-atomic-number-using-python/

def get_periodictable_list():
    atom_num = []
    symbol = []
    
    for element in periodictable.elements:
        atom_num.append(int(element.number))
        symbol.append(str(element.symbol))
    
    an2s = dict(zip(atom_num, symbol))
    s2an = dict(zip(symbol, atom_num))
    
    return an2s, s2an
