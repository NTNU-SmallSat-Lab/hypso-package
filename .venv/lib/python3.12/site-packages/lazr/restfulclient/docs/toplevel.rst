*****************
Top-level objects
*****************

Every web service has a top-level "root" object.

    >>> from lazr.restfulclient.tests.example import CookbookWebServiceClient
    >>> service = CookbookWebServiceClient()

The root object provides access to service-wide objects.

    >>> sorted(service.lp_entries)
    ['featured_cookbook']

    >>> sorted(service.lp_collections)
    ['cookbooks', 'dishes', 'recipes']


Top-level entries
=================

You can access a top-level entry through attribute access.

    >>> print service.featured_cookbook.name
    Mastering the Art of French Cooking


Top-level collections
=====================

You can access a top-level collection through attribute access.

    >>> len(service.dishes)
    3

Specific top-level collections may support key-based lookups. For
instance, the recipe collection does lookups by recipe ID. This is
custom code written for this specific web service, and it won't work
in general.

    >>> print service.recipes[1].dish.name
    Roast chicken

Looking up an object in a top-level collection triggers an HTTP
request, and if the object does not exist on the server, a KeyError is
raised.

    >>> import httplib2
    >>> httplib2.debuglevel = 1
    >>> debug_service = CookbookWebServiceClient()
    send: 'GET ...'
    ...

    >>> recipe = debug_service.recipes[4]
    send: 'GET /1.0/recipes/4 ...'
    ...

    >>> recipe = debug_service.recipes[1000]
    Traceback (most recent call last):
    ...
    KeyError: 1000

If you want to look up an object without triggering an HTTP request,
you can use parentheses instead of square brackets.

    >>> recipe = debug_service.recipes(4)
    >>> nonexistent_recipe = debug_service.recipes(1000)

    >>> sorted(recipe.lp_attributes)
    ['http_etag', 'id', 'instructions', ...]

The HTTP request, and any potential error, happens when you try to
access one of the object's properties.

    >>> print recipe.instructions
    send: 'GET /1.0/recipes/4 ...'
    ...
    Preheat oven to...

    >>> print nonexistent_recipe.instructions
    Traceback (most recent call last):
    ...
    NotFound: HTTP Error 404: Not Found
    ...

This is useful if you plan to invoke a named operation on the object
instead of accessing its properties--this can save you an HTTP request
and speed up your applicaiton.

Now, let's imagine that a top-level collection is misconfigured. We
know that the top-level collection of recipes contains objects whose
resource type is 'recipe'. But let's tell lazr.restfulclient that that
collection contains objects of type 'cookbook'.

    >>> from lazr.restfulclient.tests.example import RecipeSet
    >>> print RecipeSet.collection_of
    recipe
    >>> RecipeSet.collection_of = 'cookbook'

Looking up an object will give you something that presents the
interface of a cookbook.

    >>> not_really_a_cookbook = debug_service.recipes(2)
    >>> sorted(not_really_a_cookbook.lp_attributes)
    ['confirmed', 'copyright_date', 'cuisine'...]

But once you try to access one of the object's properties, and the
HTTP request is made...

    >>> print not_really_a_cookbook.resource_type_link
    send: 'GET /1.0/recipes/2 ...'
    ...
    http://cookbooks.dev/1.0/#recipe

...the server serves a recipe, and so the client-side object takes on
the properties of a recipe. You can only fool lazr.restfulclient up to
the point where it has real data to look at.

    >>> sorted(not_really_a_cookbook.lp_attributes)
    ['http_etag', 'id', 'instructions', ...]

    >>> print not_really_a_cookbook.instructions
    Draw, singe, stuff, and truss...

This isn't just a defense mechanism: it's a useful feature when a
top-level collection contains mixed subclasses of some superclass. For
instance, the launchpadlib library defines the 'people' collection as
containing 'team' objects, even though it also contains 'person'
objects, which expose a subset of a team's functionality. All objects
looked up in that collection start out as team objects, but once an
object's data is fetched, if it turns out to actually be a person, it
switches from the "team" interface to the "people" interface. (This
bit of hackery is necessary because WADL doesn't have an inheritance
mechanism.)

If you try to access a property based on a resource type the object
doesn't really implement, you'll get an error.

    >>> not_really_a_cookbook = debug_service.recipes(3)
    >>> sorted(not_really_a_cookbook.lp_attributes)
    ['confirmed', 'copyright_date', 'cuisine'...]
    >>> not_really_a_cookbook.cuisine
    Traceback (most recent call last):
    ...
    AttributeError: http://cookbooks.dev/1.0/recipes/3 object has no attribute 'cuisine'

Cleanup.

    >>> httplib2.debuglevel = 0
    >>> RecipeSet.collection_of = 'recipe'

Versioning
==========

By passing in a 'version' argument to the client constructor, you can
access different versions of the web service.

    >>> print service.recipes[1].self_link
    http://cookbooks.dev/1.0/recipes/1

    >>> devel_service = CookbookWebServiceClient(version="devel")
    >>> print devel_service.recipes[1].self_link
    http://cookbooks.dev/devel/recipes/1

You can also forgo the 'version' argument and pass in a service root
that incorporates a version string.

    >>> devel_service = CookbookWebServiceClient(
    ...     service_root="http://cookbooks.dev/devel/", version=None)
    >>> print devel_service.recipes[1].self_link
    http://cookbooks.dev/devel/recipes/1

Error reporting
===============

If there's an error communicating with the server, lazr.restfulclient
raises HTTPError or an appropriate subclass. The error might be a
client-side error (maybe you tried to access something that doesn't
exist) or a server-side error (maybe the server crashed due to a
bug). The string representation of the error should have enough
information to help you figure out what happened.

This example demonstrates NotFound, the HTTPError subclass used when
the server sends a 404 error For detailed information about the
different HTTPError subclasses, see tests/test_error.py.

    >>> from lazr.restfulclient.errors import HTTPError
    >>> try:
    ...    service.load("http://cookbooks.dev/")
    ... except Exception, e:
    ...     pass

    >>> raise e
    Traceback (most recent call last):
    ...
    NotFound: HTTP Error 404: Not Found
    Response headers:
    ---
    ...
    content-type: text/plain
    ...
    ---
    Response body:
    ---
    ...
    ---

    >>> print isinstance(e, HTTPError)
    True
