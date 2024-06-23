import periodictable
# https://periodictable.readthedocs.io/en/latest/guide/using.html
# https://www.geeksforgeeks.org/get-the-details-of-an-element-by-atomic-number-using-python/
from Bio.Data import IUPACData
from Bio.PDB.Polypeptide import three_to_index, index_to_one, is_aa


def get_periodictable_list(include_aa=False):
    atom_num = []
    symbol = []
    
    if include_aa:
        for three, one in IUPACData.protein_letters_3to1.items():
            atom_num.append(int(three_to_index(three.upper())) - 20)
            symbol.append(three.upper())

    for element in periodictable.elements:
        atom_num.append(int(element.number))
        symbol.append(str(element.symbol))

    an2s = dict(zip(atom_num, symbol))
    s2an = dict(zip(symbol, atom_num))
    
    return an2s, s2an


if __name__ == '__main__':
    an2s, s2an = get_periodictable_list(include_aa=True)
    print(s2an)
    for k,v in s2an.items():
        print(k, v)


# Original Amino Acid codes
# A: ALA: 0
# C: CYS: 1
# D: ASP: 2
# E: GLU: 3
# F: PHE: 4
# G: GLY: 5
# H: HIS: 6
# I: ILE: 7
# K: LYS: 8
# L: LEU: 9
# M: MET: 10
# N: ASN: 11
# P: PRO: 12
# Q: GLN: 13
# R: ARG: 14
# S: SER: 15
# T: THR: 16
# V: VAL: 17
# W: TRP: 18
# Y: TYR: 19


# Adjusted Amino Acid Codes
# ALA -20
# CYS -19
# ASP -18
# GLU -17
# PHE -16
# GLY -15
# HIS -14
# ILE -13
# LYS -12
# LEU -11
# MET -10
# ASN -9
# PRO -8
# GLN -7
# ARG -6
# SER -5
# THR -4
# VAL -3
# TRP -2
# TYR -1