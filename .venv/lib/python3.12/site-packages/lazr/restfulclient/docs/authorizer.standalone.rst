Authorizers
===========

Authorizers are objects that encapsulate knowledge about a particular
web service's authentication scheme. lazr.restfulclient includes
authorizers for common HTTP authentication schemes.

The BasicHttpAuthorizer
-----------------------

This authorizer handles HTTP Basic Auth. To test it, we'll create a
fake web service that serves some sample WADL.

    >>> import pkg_resources
    >>> wadl_string = pkg_resources.resource_string(
    ...     'wadllib.tests.data', 'launchpad-wadl.xml')

    >>> responses = { 'application/vnd.sun.wadl+xml' : wadl_string,
    ...               'application/json' : '{}' }

    >>> def sample_application(environ, start_response):
    ...     media_type = environ['HTTP_ACCEPT']
    ...     content = responses[media_type]
    ...     start_response(
    ...         '200', [('Content-type', media_type)])
    ...     return [content]


The WADL file will be protected with HTTP Basic Auth. To access it,
you'll need to provide a username of "user" and a password of
"password".

    >>> def authenticate(username, password):
    ...     """Accepts "user/password", rejects everything else.
    ...
    ...     :return: The username, if the credentials are valid.
    ...              None, otherwise.
    ...     """
    ...     if username == "user" and password == "password":
    ...         return username
    ...     return None

    >>> from lazr.authentication.wsgi import BasicAuthMiddleware
    >>> def protected_application():
    ...     return BasicAuthMiddleware(
    ...         sample_application, authenticate_with=authenticate)

Finally, we'll set up a WSGI intercept so that we can test the web
service by making HTTP requests to http://api.launchpad.dev/. (This is
the hostname mentioned in the WADL file.)

    >>> import wsgi_intercept
    >>> from wsgi_intercept.httplib2_intercept import install
    >>> install()
    >>> wsgi_intercept.add_wsgi_intercept(
    ...     'api.launchpad.dev', 80, protected_application)

With no HttpAuthorizer, a ServiceRoot can't get access to the web service.

    >>> from lazr.restfulclient.resource import ServiceRoot
    >>> client = ServiceRoot(None, "http://api.launchpad.dev/")
    Traceback (most recent call last):
    ...
    Unauthorized: HTTP Error 401: Unauthorized
    ...

We can't get access if the authorizer doesn't have the right
credentials.

    >>> from lazr.restfulclient.authorize import BasicHttpAuthorizer

    >>> bad_authorizer = BasicHttpAuthorizer("baduser", "badpassword")
    >>> client = ServiceRoot(bad_authorizer, "http://api.launchpad.dev/")
    Traceback (most recent call last):
    ...
    Unauthorized: HTTP Error 401: Unauthorized
    ...

If we provide the right credentials, we can retrieve the WADL. We'll
still get an exception, because our fake web service is too fake for
ServiceRoot--its 'service root' resource doesn't match the WADL--but
we're able to make HTTP requests without getting 401 errors.

Note that the HTTP request includes the User-Agent header, but that
that header contains no special information about the authorization
method. This will change when the authorization method is OAuth.

    >>> import httplib2
    >>> httplib2.debuglevel = 1

    >>> authorizer = BasicHttpAuthorizer("user", "password")
    >>> client = ServiceRoot(authorizer, "http://api.launchpad.dev/")
    send: 'GET / ...user-agent: lazr.restfulclient ...'
    ...


The BasicHttpAuthorizer allows you to adds proper basic auth headers to the
request, when asked to, using the username and password information it already
knows about.

    >>> headers = {}
    >>> authorizer.authorizeRequest('/', 'GET', '', headers)
    >>> headers.get('authorization')
    'Basic dXNlcjpwYXNzd29yZA=='

Teardown.

    >>> httplib2.debuglevel = 0
    >>> _ = wsgi_intercept.remove_wsgi_intercept("api.launchpad.dev", 80)


The OAuthAuthorizer
-------------------

This authorizer handles OAuth authorization. To test it, we'll protect
the sample application with a piece of OAuth middleware. The middleware
will accept only one consumer/token combination, though it will also
allow anonymous access: if you pass in an empty token and secret,
you'll get a lower level of access.

    >>> from oauth.oauth import OAuthConsumer, OAuthToken
    >>> valid_consumer = OAuthConsumer("consumer", '')
    >>> valid_token = OAuthToken("token", "secret")
    >>> empty_token = OAuthToken("", "")

Our authenticate() implementation checks against the one valid
consumer and token.

    >>> def authenticate(consumer, token, parameters):
    ...     """Accepts the valid consumer and token, rejects everything else.
    ...
    ...     :return: The consumer, if the credentials are valid.
    ...              None, otherwise.
    ...     """
    ...     if token.key == '' and token.secret == '':
    ...         # Anonymous access.
    ...         return consumer
    ...     if consumer == valid_consumer and token == valid_token:
    ...         return consumer
    ...     return None

Our data store helps the middleware look up consumer and token objects
from the information provided in a signed OAuth request.

    >>> from lazr.authentication.testing.oauth import SimpleOAuthDataStore

    >>> class AnonymousAccessDataStore(SimpleOAuthDataStore):
    ...     """A data store that will accept any consumer."""
    ...     def lookup_consumer(self, consumer):
    ...         """If there's no matching consumer, just create one.
    ...
    ...         This will let anonymous requests succeed with any
    ...         consumer key."""
    ...         consumer = super(
    ...             AnonymousAccessDataStore, self).lookup_consumer(
    ...             consumer)
    ...         if consumer is None:
    ...             consumer = OAuthConsumer(consumer, '')
    ...         return consumer

    >>> data_store = AnonymousAccessDataStore(
    ...     {valid_consumer.key : valid_consumer},
    ...     {valid_token.key : valid_token,
    ...      empty_token.key : empty_token})

Now we're ready to protect the sample_application with OAuthMiddleware,
using our authenticate() implementation and our data store.

    >>> from lazr.authentication.wsgi import OAuthMiddleware
    >>> def protected_application():
    ...     return OAuthMiddleware(
    ...         sample_application, realm="OAuth test",
    ...         authenticate_with=authenticate, data_store=data_store)
    >>> wsgi_intercept.add_wsgi_intercept(
    ...     'api.launchpad.dev', 80, protected_application)

Let's try out some clients. As you'd expect, you can't get through the
middleware with no HTTPAuthorizer at all.

    >>> from lazr.restfulclient.authorize.oauth import OAuthAuthorizer
    >>> client = ServiceRoot(None, "http://api.launchpad.dev/")
    Traceback (most recent call last):
    ...
    Unauthorized: HTTP Error 401: Unauthorized
    ...

Invalid credentials are also no help.

    >>> authorizer = OAuthAuthorizer(
    ...     valid_consumer.key, access_token=OAuthToken("invalid", "token"))
    >>> client = ServiceRoot(authorizer, "http://api.launchpad.dev/")
    Traceback (most recent call last):
    ...
    Unauthorized: HTTP Error 401: Unauthorized
    ...

But valid credentials work fine (again, up to the point at which
lazr.restfulclient runs against the limits of this simple web
service). Note that the User-Agent header mentions the consumer key.

    >>> httplib2.debuglevel = 1
    >>> authorizer = OAuthAuthorizer(
    ...     valid_consumer.key, access_token=valid_token)
    >>> client = ServiceRoot(authorizer, "http://api.launchpad.dev/")
    send: 'GET /...user-agent: lazr.restfulclient...; oauth_consumer="consumer"...'
    ...

If the OAuthAuthorizer is created with an application name as well as
a consumer key, the application name is mentioned in the User-Agent
header as well.

    >>> authorizer = OAuthAuthorizer(
    ...     valid_consumer.key, access_token=valid_token,
    ...     application_name="the app")
    >>> client = ServiceRoot(authorizer, "http://api.launchpad.dev/")
    send: 'GET /...user-agent: lazr.restfulclient...; application="the app"; oauth_consumer="consumer"...'
    ...

    >>> httplib2.debuglevel = 0

It's even possible to get anonymous access by providing an empty
access token.

    >>> authorizer = OAuthAuthorizer(
    ...     valid_consumer.key, access_token=empty_token)
    >>> client = ServiceRoot(authorizer, "http://api.launchpad.dev/")

Because of the way the AnonymousAccessDataStore (defined
earlier in the test) works, you can even get anonymous access by
specifying an OAuth consumer that's not in the server-side list of
valid consumers.

    >>> authorizer = OAuthAuthorizer(
    ...     "random consumer", access_token=empty_token)
    >>> client = ServiceRoot(authorizer, "http://api.launchpad.dev/")

A ServiceRoot object has a 'credentials' attribute which contains the
Authorizer used to authorize outgoing requests.

    >>> from lazr.restfulclient.resource import ServiceRoot
    >>> root = ServiceRoot(authorizer, "http://api.launchpad.dev/")
    >>> root.credentials
    <lazr.restfulclient.authorize.oauth.OAuthAuthorizer object...>

If you try to provide credentials with an unrecognized OAuth consumer,
you'll get an error--even if the credentials are valid. The data store
used in this test only lets unrecognized OAuth consumers through when
they request anonymous access.

    >>> authorizer = OAuthAuthorizer(
    ...     'random consumer', access_token=valid_token)
    >>> client = ServiceRoot(authorizer, "http://api.launchpad.dev/")
    Traceback (most recent call last):
    ...
    Unauthorized: HTTP Error 401: Unauthorized
    ...

    >>> authorizer = OAuthAuthorizer(
    ...     'random consumer', access_token=OAuthToken("invalid", "token"))
    >>> client = ServiceRoot(authorizer, "http://api.launchpad.dev/")
    Traceback (most recent call last):
    ...
    Unauthorized: HTTP Error 401: Unauthorized
    ...

Teardown.

    >>> _ = wsgi_intercept.remove_wsgi_intercept("api.launchpad.dev", 80)
