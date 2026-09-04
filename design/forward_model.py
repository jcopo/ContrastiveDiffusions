# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Forward model protocol extensions for Bayesian Experimental Design.

This module defines the DesignForwardModel protocol which extends the base ForwardModel
from the diffuse package with methods specific to Bayesian Experimental Design (BED).

The additional methods enable:
- Design parameter initialization and optimization
- Sequential measurement acquisition
- Adaptive experimental design strategies
"""

from typing import Protocol
from jaxtyping import Array, PRNGKeyArray
from diffuse.base_forward_model import ForwardModel, MeasurementState


class DesignForwardModel(ForwardModel, Protocol):
    """Forward model protocol with design optimization capabilities for BED.

    This protocol extends the base ForwardModel with methods needed for Bayesian
    Experimental Design applications. It enables adaptive measurement strategies
    where design parameters (e.g., sensor locations, measurement patterns) are
    optimized to maximize information gain.

    The protocol supports sequential experimental design workflows:
    1. Initialize design parameters (init_design)
    2. Convert design to measurement operator (make)
    3. Take measurements on ground truth (measure)
    4. Update measurement history (update_measurement)
    5. Optimize design to maximize EIG (via gradient-based methods)

    Example Applications:
    - Adaptive MRI k-space sampling
    - Optimal sensor placement
    - Active learning for inverse problems
    - Sequential inpainting mask design

    Attributes:
        std: Standard deviation of measurement noise (inherited from ForwardModel)
    """

    def init_design(self, rng_key: PRNGKeyArray) -> Array:
        """Initialize random design parameters.

        Creates initial design parameters for optimization. The design parametrization
        depends on the specific forward model:
        - Image inpainting: (x, y) coordinates of mask center
        - MRI sampling: k-space line indices or continuous frequencies
        - Sensor networks: (x, y, z) sensor locations

        Args:
            rng_key: JAX random key for stochastic initialization

        Returns:
            Array of design parameters with shape appropriate for the forward model.
            Should be differentiable for gradient-based optimization.

        Example:
            >>> mask = SquareMask(size=7, img_shape=(28, 28, 1))
            >>> design = mask.init_design(jax.random.PRNGKey(0))
            >>> design.shape
            (2,)  # (x, y) coordinates
        """
        ...

    def make(self, design: Array) -> Array:
        """Convert design parameters to measurement operator/mask.

        Creates a differentiable measurement operator from design parameters.
        This operator defines which parts of the signal are observed.

        For CoDiff optimization, this function must be:
        1. Differentiable w.r.t. design parameters
        2. Output values in [0, 1] for soft masking (enables gradients)
        3. Interpretable as measurement probabilities or weights

        Args:
            design: Design parameters from init_design or optimization

        Returns:
            Measurement operator/mask with same spatial dimensions as the signal.
            Typically has values in [0, 1] where 1 = fully observed, 0 = unobserved.

        Example:
            >>> design = jnp.array([14.0, 14.0])  # Center of 28x28 image
            >>> mask = mask_model.make(design)
            >>> mask.shape
            (28, 28, 1)
        """
        ...

    def update_measurement(
        self,
        measurement_state: MeasurementState,
        new_measurement: Array,
        design: Array,
    ) -> MeasurementState:
        """Update measurement history with new sequential observations.

        Accumulates measurements across sequential experimental design iterations.
        This enables:
        1. Conditional sampling on all past observations
        2. Non-redundant design optimization (avoid re-measuring same regions)
        3. Sequential information gain calculation

        The update typically involves:
        - Adding new_measurement to existing observations (y)
        - Updating mask_history to track which regions have been measured

        Args:
            measurement_state: Current state containing (y, mask_history)
            new_measurement: Fresh measurement from current design
            design: Design parameters that generated new_measurement

        Returns:
            Updated MeasurementState with:
            - y: Combined observations (past + new)
            - mask_history: Updated measurement coverage

        Example:
            >>> # Initial state: no measurements
            >>> state = mask_model.init(key)
            >>>
            >>> # Take first measurement
            >>> state = mask_model.update_measurement(state, ground_truth, design1)
            >>>
            >>> # Take second measurement (different design)
            >>> state = mask_model.update_measurement(state, ground_truth, design2)
            >>> # state.y now contains both measurements
        """
        ...

    def init(self, rng_key: PRNGKeyArray) -> MeasurementState:
        """Initialize empty measurement state for sequential BED.

        Creates initial MeasurementState with:
        - y: Zero array (no observations yet)
        - mask_history: Zero array (nothing measured yet)

        Args:
            rng_key: JAX random key (may be used for noise initialization)

        Returns:
            Initial MeasurementState with empty observations and mask history

        Example:
            >>> state = mask_model.init(jax.random.PRNGKey(0))
            >>> jnp.all(state.y == 0)
            True
            >>> jnp.all(state.mask_history == 0)
            True
        """
        ...

    # Inherited from ForwardModel protocol:
    #
    # def apply(self, img: Array, measurement_state: MeasurementState) -> Array:
    #     """Apply forward measurement operator (inherited from ForwardModel)."""
    #     ...
    #
    # def restore(self, img: Array, measurement_state: MeasurementState) -> Array:
    #     """Apply adjoint operator for guidance (inherited from ForwardModel)."""
    #     ...
