****************
People and Teams
****************

The Launchpad web service, like Launchpad itself, exposes a unified
interface to people and teams.  In other words, people and teams
occupy the same namespace.  You treat people and teams as the same
type of object, and need to inspect the object to know whether you're
dealing with a person or a team.


People
======

You can access Launchpad people through the web service interface.
The list of people is available from the service root.

    >>> from launchpadlib.testing.helpers import salgado_with_full_permissions
    >>> launchpad = salgado_with_full_permissions.login()
    >>> people = launchpad.people

The list of people is not fetched until you actually use data.

    >>> print(people._wadl_resource.representation)
    None

    >>> len(people)
    4

    >>> print(people._wadl_resource.representation)
    {...}

The 'me' attribute is also available from the service root. It's a
quick way to get a reference to your own user account.

    >>> me = launchpad.me
    >>> print(me.name)
    salgado

You can find a person by name.

    >>> salgado = launchpad.people['salgado']
    >>> print(salgado.name)
    salgado
    >>> print(salgado.display_name)
    Guilherme Salgado
    >>> salgado.is_team
    False

But if no person by that name is registered, you get the expected KeyError.

    >>> launchpad.people['not-a-registered-person']
    Traceback (most recent call last):
    ...
    KeyError: 'not-a-registered-person'

It's not possible to slice a single person from the top-level
collection of people. launchpadlib will try to use the value you pass
in as a person's name, which will almost always fail.

    >>> launchpad.people[1]
    Traceback (most recent call last):
    ...
    KeyError: 1

You can find a person by email.

    >>> email = salgado.preferred_email_address.email
    >>> salgado = launchpad.people.getByEmail(email=email)
    >>> print(salgado.name)
    salgado

Besides a name and a display name, a person has many other attributes that you
can read.

    XXX 05-Jun-2008 BarryWarsaw Some of these attributes are links to further
    collections and are not yet tested.  Tests will be added in future
    branches.

    >>> salgado.karma
    0
    >>> print(salgado.homepage_content)
    None
    >>> #salgado.mugshot
    >>> #salgado.languages
    >>> salgado.hide_email_addresses
    False
    >>> salgado.date_created
    datetime.datetime(2005, 6, 6, 8, 59, 51, 596025, ...)
    >>> print(salgado.time_zone)
    UTC
    >>> salgado.is_valid
    True
    >>> #salgado.wiki_names
    >>> #salgado.irc_nicknames
    >>> #salgado.jabber_ids
    >>> #salgado.team_memberships
    >>> #salgado.open_membership_invitations
    >>> #salgado.teams_participated_in
    >>> #salgado.teams_indirectly_participated_in
    >>> #salgado.confirmed_email_addresses
    >>> #salgado.preferred_email_address
    >>> print(salgado.mailing_list_auto_subscribe_policy)
    Ask me when I join a team
    >>> print(salgado.visibility)
    Public


Teams
=====

You also access teams using the same interface.

    >>> team = launchpad.people['ubuntu-team']
    >>> print(team.name)
    ubuntu-team
    >>> print(team.display_name)
    Ubuntu Team
    >>> team.is_team
    True

Regular people have team attributes, but they're not used.

    >>> print(salgado.team_owner)
    None

You can find out how a person has membership in a team.

    # XXX: salgado, 2008-08-01: Commented because method has been Unexported;
    # it should be re-enabled after the operation is exported again.
    # >>> path = salgado.findPathToTeam(
    # ...     team=launchpad.people['mailing-list-experts'])
    # >>> [team.name for team in path]
    # [u'admins', u'mailing-list-experts']

You can create a new team through the web interface.  The simplest case of
this requires only the new team's name, owner and display name.

    >>> launchpad.people['bassists']
    Traceback (most recent call last):
    ...
    KeyError: 'bassists'

    >>> bassists = launchpad.people.newTeam(
    ...     name='bassists', display_name='Awesome Rock Bass Players')
    >>> print(bassists.name)
    bassists
    >>> print(bassists.display_name)
    Awesome Rock Bass Players
    >>> bassists.is_team
    True

And of course, that team is now accessible directly.

    >>> bassists = launchpad.people['bassists']
    >>> print(bassists.name)
    bassists
    >>> print(bassists.display_name)
    Awesome Rock Bass Players

You cannot create the same team twice.

    >>> launchpad.people.newTeam(name='bassists', display_name='Bass Gods')
    Traceback (most recent call last):
    ...
    lazr.restfulclient.errors.BadRequest: HTTP Error 400: Bad Request
    ...

Actually, the exception contains other useful information.

    >>> from launchpadlib.errors import HTTPError
    >>> try:
    ...     launchpad.people.newTeam(
    ...         name='bassists', display_name='Bass Gods')
    ... except HTTPError as e:
    ...     error = e
    >>> error.response['status']
    '400'
    >>> print(error.content.decode())
    name: bassists is already in use by another person or team.

Besides a name and a display name, a team has many other attributes that you
can read.

    >>> bassists.karma
    0
    >>> print(bassists.homepage_content)
    None
    >>> bassists.hide_email_addresses
    False
    >>> bassists.date_created
    datetime.datetime(...)
    >>> print(bassists.time_zone)
    UTC
    >>> bassists.is_valid
    True
    >>> #bassists.team_memberships
    >>> #bassists.open_membership_invitations
    >>> #bassists.teams_participated_in
    >>> #bassists.teams_indirectly_participated_in
    >>> #bassists.confirmed_email_addresses
    >>> #bassists.team_owner
    >>> #bassists.preferred_email_address
    >>> #bassists.members
    >>> #bassists.admins
    >>> #bassists.participants
    >>> #bassists.deactivated_members
    >>> #bassists.expired_members
    >>> #bassists.invited_members
    >>> #bassists.member_memberships
    >>> #bassists.proposed_members
    >>> print(bassists.visibility)
    Public
    >>> print(bassists.team_description)
    None
    >>> print(bassists.subscription_policy)
    Moderated Team
    >>> print(bassists.renewal_policy)
    invite them to apply for renewal
    >>> print(bassists.default_membership_period)
    None
    >>> print(bassists.default_renewal_period)
    None
