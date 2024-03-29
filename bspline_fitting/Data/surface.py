import pickle
from . import abstract, evaluators, operations, tessellate, utilities
from . import _utilities as utl
from .exceptions import GeomdlException


class Surface(abstract.Surface):
    """ Data storage and evaluation class for B-spline (non-rational) surfaces.

    This class provides the following properties:

    * :py:attr:`type` = spline
    * :py:attr:`id`
    * :py:attr:`order_u`
    * :py:attr:`order_v`
    * :py:attr:`degree_u`
    * :py:attr:`degree_v`
    * :py:attr:`knotvector_u`
    * :py:attr:`knotvector_v`
    * :py:attr:`ctrlpts`
    * :py:attr:`ctrlpts_size_u`
    * :py:attr:`ctrlpts_size_v`
    * :py:attr:`ctrlpts2d`
    * :py:attr:`delta`
    * :py:attr:`delta_u`
    * :py:attr:`delta_v`
    * :py:attr:`sample_size`
    * :py:attr:`sample_size_u`
    * :py:attr:`sample_size_v`
    * :py:attr:`bbox`
    * :py:attr:`name`
    * :py:attr:`dimension`
    * :py:attr:`vis`
    * :py:attr:`evaluator`
    * :py:attr:`tessellator`
    * :py:attr:`rational`
    * :py:attr:`trims`

    The following code segment illustrates the usage of Surface class:

    .. code-block:: python
        :linenos:

        from geomdl import BSpline

        # Create a BSpline surface instance (Bezier surface)
        surf = BSpline.Surface()

        # Set degrees
        surf.degree_u = 3
        surf.degree_v = 2

        # Set control points
        control_points = [[0, 0, 0], [0, 4, 0], [0, 8, -3],
                          [2, 0, 6], [2, 4, 0], [2, 8, 0],
                          [4, 0, 0], [4, 4, 0], [4, 8, 3],
                          [6, 0, 0], [6, 4, -3], [6, 8, 0]]
        surf.set_ctrlpts(control_points, 4, 3)

        # Set knot vectors
        surf.knotvector_u = [0, 0, 0, 0, 1, 1, 1, 1]
        surf.knotvector_v = [0, 0, 0, 1, 1, 1]

        # Set evaluation delta (control the number of surface points)
        surf.delta = 0.05

        # Get surface points (the surface will be automatically evaluated)
        surface_points = surf.evalpts

    **Keyword Arguments:**

    * ``precision``: number of decimal places to round to. *Default: 18*
    * ``normalize_kv``: activates knot vector normalization. *Default: True*
    * ``find_span_func``: sets knot span search implementation. *Default:* :func:`.helpers.find_span_linear`
    * ``insert_knot_func``: sets knot insertion implementation. *Default:* :func:`.operations.insert_knot`
    * ``remove_knot_func``: sets knot removal implementation. *Default:* :func:`.operations.remove_knot`

    Please refer to the :py:class:`.abstract.Surface()` documentation for more details.
    """
    # __slots__ = ('_insert_knot_func', '_remove_knot_func', '_control_points2D')

    def __init__(self, **kwargs):
        super(Surface, self).__init__(**kwargs)
        self._evaluator = evaluators.SurfaceEvaluator(find_span_func=self._span_func)
        self._tsl_component = tessellate.TriangularTessellate()
        self._control_points2D = self._init_array()  # control points, 2-D array [u][v]
        self._insert_knot_func = kwargs.get('insert_knot_func', operations.insert_knot)
        self._remove_knot_func = kwargs.get('remove_knot_func', operations.remove_knot)

    @property
    def ctrlpts2d(self):
        """ 2-dimensional array of control points.

        The getter returns a tuple of 2D control points (weighted control points + weights if NURBS) in *[u][v]* format.
        The rows of the returned tuple correspond to v-direction and the columns correspond to u-direction.

        The following example can be used to traverse 2D control points:

        .. code-block:: python
            :linenos:

            # Create a BSpline surface
            surf_bs = BSpline.Surface()

            # Do degree, control points and knot vector assignments here

            # Each u includes a row of v values
            for u in surf_bs.ctrlpts2d:
                # Each row contains the coordinates of the control points
                for v in u:
                    print(str(v))  # will be something like (1.0, 2.0, 3.0)

            # Create a NURBS surface
            surf_nb = NURBS.Surface()

            # Do degree, weighted control points and knot vector assignments here

            # Each u includes a row of v values
            for u in surf_nb.ctrlpts2d:
                # Each row contains the coordinates of the weighted control points
                for v in u:
                    print(str(v))  # will be something like (0.5, 1.0, 1.5, 0.5)


        When using **NURBS.Surface** class, the output of :py:attr:`~ctrlpts2d` property could be confusing since,
        :py:attr:`~ctrlpts` always returns the unweighted control points, i.e. :py:attr:`~ctrlpts` property returns 3D
        control points all divided by the weights and you can use :py:attr:`~weights` property to access the weights
        vector, but :py:attr:`~ctrlpts2d` returns the weighted ones plus weights as the last element.
        This difference is intentionally added for compatibility and interoperability purposes.

        To explain this situation in a simple way;

        * If you need the weighted control points directly, use :py:attr:`~ctrlpts2d`
        * If you need the control points and the weights separately, use :py:attr:`~ctrlpts` and :py:attr:`~weights`

        .. note::

            Please note that the setter doesn't check for inconsistencies and using the setter is not recommended.
            Instead of the setter property, please use :func:`.set_ctrlpts()` function.

        Please refer to the `wiki <https://github.com/orbingol/NURBS-Python/wiki/Using-Python-Properties>`_ for details
        on using this class member.

        :getter: Gets the control points as a 2-dimensional array in [u][v] format
        :setter: Sets the control points as a 2-dimensional array in [u][v] format
        :type: list
        """
        return self._control_points2D

    @ctrlpts2d.setter
    def ctrlpts2d(self, value):
        if not isinstance(value, (list, tuple)):
            raise ValueError("The input must be a list or tuple")

        # Clean up the surface and control points
        self.reset(evalpts=True, ctrlpts=True)

        # Assume that the user has prepared the lists correctly
        size_u = len(value)
        size_v = len(value[0])

        # Estimate dimension by checking the size of the first element
        self._dimension = len(value[0][0])

        # Make sure that all numbers are float type
        ctrlpts = [[] for _ in range(size_u * size_v)]
        for u in range(size_u):
            for v in range(size_v):
                idx = v + (size_v * u)
                ctrlpts[idx] = [float(coord) for coord in value[u][v]]

        # Set control points
        self.set_ctrlpts(ctrlpts, size_u, size_v)

    def set_ctrlpts(self, ctrlpts, *args, **kwargs):
        """ Sets the control points and checks if the data is consistent.

        This method is designed to provide a consistent way to set control points whether they are weighted or not.
        It directly sets the control points member of the class, and therefore it doesn't return any values.
        The input will be an array of coordinates. If you are working in the 3-dimensional space, then your coordinates
        will be an array of 3 elements representing *(x, y, z)* coordinates.

        This method also generates 2D control points in *[u][v]* format which can be accessed via :py:attr:`~ctrlpts2d`.

        .. note::

            The v index varies first. That is, a row of v control points for the first u value is found first.
            Then, the row of v control points for the next u value.

        :param ctrlpts: input control points as a list of coordinates
        :type ctrlpts: list
        """
        # Call parent function
        super(Surface, self).set_ctrlpts(ctrlpts, *args, **kwargs)

        # Generate a 2-dimensional list of control points
        array_init2d = kwargs.get('array_init2d', [[[] for _ in range(args[1])] for _ in range(args[0])])
        ctrlpts_float2d = array_init2d
        for i in range(0, self.ctrlpts_size_u):
            for j in range(0, self.ctrlpts_size_v):
                ctrlpts_float2d[i][j] = self._control_points[j + (i * self.ctrlpts_size_v)]

        # Set the new 2-dimension control points
        self._control_points2D = ctrlpts_float2d

    def reset(self, **kwargs):
        """ Resets control points and/or evaluated points.

        Keyword Arguments:
            * ``evalpts``: if True, then resets evaluated points
            * ``ctrlpts`` if True, then resets control points

        """
        # Call parent function
        super(Surface, self).reset(**kwargs)

        # Reset ctrlpts2d
        reset_ctrlpts = kwargs.get('ctrlpts', False)
        if reset_ctrlpts:
            self._control_points2D = self._init_array()

    def save(self, file_name):
        """ Saves the surface as a pickled file.

        .. deprecated:: 5.2.4

            Use :func:`.exchange.export_json()` instead.

        :param file_name: name of the file to be saved
        :type file_name: str
        """
        return None

    def load(self, file_name):
        """ Loads the surface from a pickled file.

        .. deprecated:: 5.2.4

            Use :func:`.exchange.import_json()` instead.

        :param file_name: name of the file to be loaded
        :type file_name: str
        """
        return None

    def transpose(self):
        """ Transposes the surface by swapping u and v parametric directions. """
        operations.transpose(self, inplace=True)
        self.reset(evalpts=True)

    def evaluate(self, **kwargs):
        """ Evaluates the surface.

        The evaluated points are stored in :py:attr:`evalpts` property.

        Keyword arguments:
            * ``start_u``: start parameter on the u-direction
            * ``stop_u``: stop parameter on the u-direction
            * ``start_v``: start parameter on the v-direction
            * ``stop_v``: stop parameter on the v-direction

        The ``start_u``, ``start_v`` and ``stop_u`` and ``stop_v`` parameters allow evaluation of a surface segment
        in the range  *[start_u, stop_u][start_v, stop_v]* i.e. the surface will also be evaluated at the ``stop_u``
        and ``stop_v`` parameter values.

        The following examples illustrate the usage of the keyword arguments.

        .. code-block:: python
            :linenos:

            # Start evaluating in range u=[0, 0.7] and v=[0.1, 1]
            surf.evaluate(stop_u=0.7, start_v=0.1)

            # Start evaluating in range u=[0, 1] and v=[0.1, 0.3]
            surf.evaluate(start_v=0.1, stop_v=0.3)

            # Get the evaluated points
            surface_points = surf.evalpts

        """
        # Call parent method
        super(Surface, self).evaluate(**kwargs)

        # Find evaluation start and stop parameter values
        start_u = kwargs.get('start_u', self.knotvector_u[self.degree_u])
        stop_u = kwargs.get('stop_u', self.knotvector_u[-(self.degree_u + 1)])
        start_v = kwargs.get('start_v', self.knotvector_v[self.degree_v])
        stop_v = kwargs.get('stop_v', self.knotvector_v[-(self.degree_v + 1)])

        # Check parameters
        if self._kv_normalize:
            if not utilities.check_params([start_u, stop_u, start_v, stop_v]):
                raise GeomdlException("Parameters should be between 0 and 1")

        # Clean up the surface points
        self.reset(evalpts=True)

        # Evaluate and cache
        self._eval_points = self._evaluator.evaluate(self.data,
                                                     start=(start_u, start_v),
                                                     stop=(stop_u, stop_v))

    def evaluate_single(self, param):
        """ Evaluates the surface at the input (u, v) parameter pair.

        :param param: parameter pair (u, v)
        :type param: list, tuple
        :return: evaluated surface point at the given parameter pair
        :rtype: list
        """
        # Call parent method
        super(Surface, self).evaluate_single(param)

        # Evaluate the surface point
        pt = self._evaluator.evaluate(self.data, start=param, stop=param)

        return pt[0]

    def evaluate_list(self, param_list):
        """ Evaluates the surface for a given list of (u, v) parameters.

        :param param_list: list of parameter pairs (u, v)
        :type param_list: list, tuple
        :return: evaluated surface point at the input parameter pairs
        :rtype: tuple
        """
        # Call parent method
        super(Surface, self).evaluate_list(param_list)

        # Evaluate (u,v) list
        res = []
        for prm in param_list:
            if self._kv_normalize:
                if utilities.check_params(prm):
                    res.append(self.evaluate_single(prm))
            else:
                res.append(self.evaluate_single(prm))
        return res

    # Evaluates n-th order surface derivatives at the given (u,v) parameter
    def derivatives(self, u, v, order=0, **kwargs):
        """ Evaluates n-th order surface derivatives at the given (u, v) parameter pair.

        * SKL[0][0] will be the surface point itself
        * SKL[0][1] will be the 1st derivative w.r.t. v
        * SKL[2][1] will be the 2nd derivative w.r.t. u and 1st derivative w.r.t. v

        :param u: parameter on the u-direction
        :type u: float
        :param v: parameter on the v-direction
        :type v: float
        :param order: derivative order
        :type order: integer
        :return: A list SKL, where SKL[k][l] is the derivative of the surface S(u,v) w.r.t. u k times and v l times
        :rtype: list
        """
        # Call parent method
        super(Surface, self).derivatives(u=u, v=v, order=order, **kwargs)

        # Evaluate and return the derivatives
        return self._evaluator.derivatives(self.data, parpos=(u, v), deriv_order=order)

    def insert_knot(self, u=None, v=None, **kwargs):
        """ Inserts knot(s) on the u- or v-directions

        Keyword Arguments:
            * ``num_u``: Number of knot insertions on the u-direction. *Default: 1*
            * ``num_v``: Number of knot insertions on the v-direction. *Default: 1*

        :param u: knot to be inserted on the u-direction
        :type u: float
        :param v: knot to be inserted on the v-direction
        :type v: float
        """
        # Check all parameters are set before the evaluation
        self._check_variables()

        # Check if the parameter values are correctly defined
        if self._kv_normalize:
            if not utilities.check_params([u, v]):
                raise GeomdlException("Parameters should be between 0 and 1")

        # Get keyword arguments
        num_u = kwargs.get('num_u', 1)  # number of knot insertions on the u-direction
        num_v = kwargs.get('num_v', 1)  # number of knot insertions on the v-direction
        check_num = kwargs.get('check_r', True)  # Enables/disables number of knot insertions checking

        # Insert knots
        try:
            self._insert_knot_func(self, [u, v], [num_u, num_v], check_num=check_num)
        except GeomdlException as e:
            print(e)
            return

        # Evaluate surface again if it has already been evaluated before knot insertion
        if check_num and self._eval_points:
            self.evaluate()

    def remove_knot(self, u=None, v=None, **kwargs):
        """ Inserts knot(s) on the u- or v-directions

        Keyword Arguments:
            * ``num_u``: Number of knot removals on the u-direction. *Default: 1*
            * ``num_v``: Number of knot removals on the v-direction. *Default: 1*

        :param u: knot to be removed on the u-direction
        :type u: float
        :param v: knot to be removed on the v-direction
        :type v: float
        """
        # Check all parameters are set before the evaluation
        self._check_variables()

        # Check if the parameter values are correctly defined
        if self._kv_normalize:
            if not utilities.check_params([u, v]):
                raise GeomdlException("Parameters should be between 0 and 1")

        # Get keyword arguments
        num_u = kwargs.get('num_u', 1)  # number of knot removals on the u-direction
        num_v = kwargs.get('num_v', 1)  # number of knot removals on the v-direction
        check_num = kwargs.get('check_r', True)  # can be set to False when the caller checks number of removals

        # Remove knots
        try:
            self._remove_knot_func(self, [u, v], [num_u, num_v], check_num=check_num)
        except GeomdlException as e:
            print(e)
            return

        # Evaluate curve again if it has already been evaluated before knot removal
        if check_num and self._eval_points:
            self.evaluate()

    def tangent(self, parpos, **kwargs):
        """ Evaluates the tangent vectors of the surface at the given parametric position(s).

        .. deprecated: 5.3.0

            Please use :func:`operations.tangent` instead.

        :param parpos: parametric position(s) where the evaluation will be executed
        :type parpos: list or tuple
        :return: an array containing "point" and "vector"s on u- and v-directions, respectively
        :rtype: tuple
        """
        return tuple()

    def normal(self, parpos, **kwargs):
        """ Evaluates the normal vector of the surface at the given parametric position(s).

        .. deprecated: 5.3.0

            Please use :func:`operations.normal` instead.

        :param parpos: parametric position(s) where the evaluation will be executed
        :type parpos: list or tuple
        :return: an array containing "point" and "vector" pairs
        :rtype: tuple
        """
        return tuple()


