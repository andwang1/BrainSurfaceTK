
import mod_wsgi.server

resources = None

handler = mod_wsgi.server.ResourceHandler(resources)

reload_required = handler.reload_required
handle_request = handler.handle_request

