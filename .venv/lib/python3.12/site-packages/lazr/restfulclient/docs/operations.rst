****************
Named operations
****************

Entries and collections support named operations: one-off
functionality that's been given a name and a set of parameters.

    >>> from lazr.restfulclient.tests.example import CookbookWebServiceClient
    >>> service = CookbookWebServiceClient()

Arguments to named operations are automatically converted to JSON for
transmission over the wire. Here's a named operation that takes a
boolean argument.

    >>> [recipe for recipe in service.cookbooks.find_recipes(
    ...      search="Chicken", vegetarian=True)]
    []

Strings that happen to be numbers are handled properly. Here, if "1.234"
were converted into a number at any point in the chain, the 'search'
operation on the server wouldn't know how to handle it and the request
would fail.

    >>> [people for people in service.cookbooks.find_recipes(search="1.234")]
    []

Counts
------

Named operations that return a collection return either a link that
lazr.restfulclient can follow to get the size of the collection or
(under some circumstances) the length itself.  The len() function
hides this indirection from the end-user.

    >>> results = service.cookbooks.find_recipes(search="Chicken")
    >>> print len(results)
    0

Special data types
------------------

lazr.restfulclient uses some data types that don't directly correspond
to JSON data types. These data types can be used in named
operations. For instance, a named operation can take a date or
datetime object as one of its arguments.

    >>> import datetime
    >>> date = datetime.datetime(1994, 1, 1)
    >>> cookbook = service.cookbooks.create(
    ...     name="New cookbook", cuisine="General",
    ...     copyright_date=date, price=1.23, last_printing=date)
    >>> print cookbook.name
    New cookbook

A named operation can take an entry as one of its arguments.
lazr.restfulclient lets you pass in the actual entry as the argument
value.

    >>> dish = service.recipes[1].dish
    >>> cookbook = service.recipes[1].cookbook
    >>> print cookbook.find_recipe_for(dish=dish)
    http://cookbooks.dev/1.0/recipes/1

A named operation can take binary data as one of its arguments.

    >>> cookbook.replace_cover(cover="\x00\xe2\xe3")
    >>> cookbook.cover.open().read()
    '\x00\xe2\xe3'

A named operation that returns a null value corresponds to a Python
value of None.

    >>> dish = service.recipes[4].dish
    >>> print cookbook.find_recipe_for(dish=dish)
    None

A named operation may change the resource's location so we get a 301 response
with the new URL.

    >>> from urllib import quote
    >>> from lazr.restful.testing.webservice import WebServiceCaller
    >>> webservice = WebServiceCaller(domain='cookbooks.dev')
    >>> url = quote("/cookbooks/New cookbook")
    >>> print webservice.named_post(url, 'make_more_interesting')
    HTTP/1.1 301 Moved Permanently
    ...
    Location: http://cookbooks.dev/devel/cookbooks/The%20New%20New%20cookbook
    ...

JSON-encoding
-------------

lazr.restfulclient encodes most arguments (even string arguments) as
JSON before sending them over the wire. This way, a named operation
that takes a string argument can take a string that looks like a JSON
object without causing confusion.

    >>> cookbooks = service.cookbooks.find_for_cuisine(cuisine="General")
    >>> len([cookbook for cookbook in cookbooks]) > 0
    True

    >>> cookbook = service.cookbooks.create(
    ...    name="null", cuisine="General",
    ...    copyright_date=date, price=1.23, last_printing=date)
    >>> cookbook.name
    u'null'

    >>> cookbook = service.cookbooks.create(
    ...     name="4.56", cuisine="General",
    ...     copyright_date=date, price=1.23, last_printing=date)
    >>> cookbook.name
    u'4.56'

    >>> cookbook = service.cookbooks.create(
    ...     name='"foo"', cuisine="General",
    ...     copyright_date=date, price=1.23, last_printing=date)
    >>> cookbook.name
    u'"foo"'

A named operation that takes a non-string object (such as a float)
will not accept a string that's the JSON representation of the
object.

    >>> try:
    ...     service.cookbooks.create(
    ...         name="Yet another 1.23 cookbook", cuisine="General",
    ...         copyright_date=date, last_printing=date, price="1.23")
    ... except Exception, e:
    ...     print e.content
    price: got 'unicode', expected float, int: u'1.23'

Named operations on collections don't fetch the collections
-----------------------------------------------------------

If you invoke a named operation on a collection, the only HTTP request
made is the one for the named operation. You don't have to get a
representation of the collection to invoke the operation.

    >>> import httplib2
    >>> httplib2.debuglevel = 1
    >>> service = CookbookWebServiceClient()
    send: ...
    ...

    >>> print service.cookbooks.find_recipes(
    ...      search="Chicken", vegetarian=True)
    send: 'GET /1.0/cookbooks?...vegetarian=true...'
    ...

Cleanup.

    >>> httplib2.debuglevel = None
