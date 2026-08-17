"""SoulCore runtime package.

Runtime services are imported from their owning feature packages.  Keeping the
package root intentionally small prevents importing the complete plugin graph
when a single contract or utility is needed.
"""
