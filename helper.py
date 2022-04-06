# -*- coding: utf-8 -*-

def getData(From, To, Offset, df_in):
    """Helper function to get data from datetime intervals"""
    df = df_in.copy()
    df = df[df.index >= From]
    df = df[df.index <= To]
    hist = []

    walker = From
    i = 0
    while(walker < To):
        df_temp = df.copy()
        df_temp = df_temp[df_temp.index >= walker]
        df_temp = df_temp[df_temp.index < walker + Offset]
        zone_list = [len(df_temp[df_temp.Zona == i].index)
                     for i in range(0, 6)]
        hist.append([walker, len(df_temp.index), zone_list])
        i += 1
        walker += Offset
    return hist
