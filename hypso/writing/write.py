def set_or_create_attr(var, attr_name, attr_value) -> None:
    """
    Set or create an attribute on ".nc" file.

    :param var: Variable on to which assign the attribute
    :param attr_name: Attribute name
    :param attr_value: Attribute value

    :return: No return value
    """

    if attr_name in var.ncattrs():
        var.setncattr(attr_name, attr_value)
        return
    var.UnusedNameAttribute = attr_value
    var.renameAttribute("UnusedNameAttribute", attr_name)
    return