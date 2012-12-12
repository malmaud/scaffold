"""
Scaffold module.
"""

# Set up the namespace for when the package is imported.
from runner import History
import scaffold
import datasources
import helpers
import demo_coin
import storage

#Expose common symbols to the top level, for convenience and to keep a stable external interface in case internal package organization changes.
from scaffold import State, Chain, retrieve_cloud_history
from storage import CloudStore
from datasources import FiniteMixture


