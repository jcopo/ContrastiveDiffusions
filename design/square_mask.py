# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""SquareMask with Bayesian Experimental Design capabilities.

This module extends the base SquareMask from the diffuse package with methods
needed for adaptive experimental design and sequential measurement acquisition.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from jaxtyping import Array, PRNGKeyArray
from diffuse.examples.mnist.forward_model import SquareMask
from diffuse.base_forward_model import MeasurementState
from design.forward_model import DesignForwardModel


@dataclass
class SquareMaskDesign(SquareMask, DesignForwardModel):
    """SquareMask with design optimization capabilities for BED."""

    def init_design(self, rng_key: PRNGKeyArray) -> Array:
        """Initialize random (x, y) coordinates for square mask center"""
        height, width, *_ = self.img_shape
        half = self.size / 2
        return jax.random.uniform(rng_key, (2,), minval=half, maxval=float(height) - half)

    def update_measurement(
        self,
        measurement_state: MeasurementState,
        new_measurement: Array,
        design: Array,
    ) -> MeasurementState:
        """Update measurement history with new sequential observation."""
        new_mask = self.make(design)
        # Compute only the newly observed region (avoid double-counting)
        new_part_mask = new_mask * (1 - measurement_state.mask_history)
        # Add new measurement only in newly observed region
        joint_y = measurement_state.y + new_measurement * new_part_mask
        # Update mask history with union of old and new masks
        mask_history = measurement_state.mask_history + new_part_mask

        return MeasurementState(y=joint_y, mask_history=mask_history)

    def init(self, rng_key: PRNGKeyArray) -> MeasurementState:
        """Initialize empty measurement state for sequential BED."""
        y = jnp.zeros(self.img_shape)
        # Ensure mask_history has same spatial dims as make() output
        mask_history = jnp.zeros((self.img_shape[0], self.img_shape[1], 1))
        return MeasurementState(y=y, mask_history=mask_history)
