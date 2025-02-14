from pycpd import RigidRegistration, DeformableRegistration
from pycpd import utility


def compare_registration_methods(X, Y, methods=None):
    """
    Compare different registration methods.

    Parameters:
        X: Target point cloud
        Y: Source point cloud
        methods: Dict of registration methods to compare
                e.g., {'rigid': RigidRegistration, 'deformable': DeformableRegistration}
    """
    if methods is None:
        methods = {
            'rigid': RigidRegistration,
            'deformable': DeformableRegistration
        }

    results = {}
    for name, RegistrationClass in methods.items():
        reg = RegistrationClass(**{'X': X, 'Y': Y})
        TY, params = reg.register()
        results[name] = {
            'transformed_points': TY,
            'parameters': params,
            'metrics': utility.calculate_registration_metrics(X, TY)
        }

    return results