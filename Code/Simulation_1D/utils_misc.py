def fill_dict(mydict,default_dict):
    '''
    For any key in default_dict that does not appear in mydict add the key

    Parameters
    ----------
    mydict : 
        Dictionary to modify

    default_dict:
        Dictionary with default answers

    Returns
    -------
    mydict : dictionary
        The dictionary with added keys
        

    '''
    if mydict is None:
        mydict = {}
    for key in default_dict.keys():
        if key not in mydict.keys():
            mydict[key] = default_dict[key]
    return mydict

def pretty_string_dict(myDict):
    '''
    Used for printing a dictionary in an easy-to-read way
    Parameters
    ----------
    myDict : dictionary
        Generic
    myStr : pair of keys etc
    '''
    myStr = ''
    for keys,values in myDict.items():
        myStr = myStr + '%s: %s\n'%(str(keys),str(values))
    return myStr
