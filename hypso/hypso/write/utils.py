def set_or_create_attr(var, attr_name, attr_value) -> None:
    """
    Set or create an attribute on ".nc" file.
    Handles boolean values by converting to integers (NetCDF doesn't support bool).

    :param var: Variable on which to assign the attribute
    :param attr_name: Attribute name
    :param attr_value: Attribute value

    :return: No return value
    """
    # Convert boolean to int (NetCDF doesn't support bool)
    if isinstance(attr_value, bool):
        attr_value = 1 if attr_value else 0
    
    # Convert None to empty string
    elif attr_value is None:
        attr_value = ""
    
    if attr_name in var.ncattrs():
        var.setncattr(attr_name, attr_value)
        return
    var.UnusedNameAttribute = attr_value
    var.renameAttribute("UnusedNameAttribute", attr_name)
    return