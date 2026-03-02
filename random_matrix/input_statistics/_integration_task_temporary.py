
    def _get_covariance_integrand(
        self,
        wave_block_one: str,
        wave_block_two: str,
        block_one: str,
        block_two: str,
        is_eight_dimensions: bool = False,
        is_pseudo_covariance: bool = False,
    ) -> MathematicalFunction:
        """Get the integrand for covariance calculations. Note that this does
        not include mode weights. We choose to add those in at the end."""

        covariance_a_matrix = (
            self.medium_statistics.get_covariance_a_matrix()
            if not is_pseudo_covariance
            else self.medium_statistics.get_pseudo_covariance_a_matrix()
        )
        k = self.medium_parameters.k
        L = self.medium_parameters.L

        ki_z_factor = -1.0 if block_one in {"r2", "t2"} else 1.0
        kj_z_factor = -1.0 if block_one in {"r", "t2"} else 1.0
        ku_z_factor = -1.0 if block_two in {"r2", "t2"} else 1.0
        kv_z_factor = -1.0 if block_two in {"r", "t2"} else 1.0
        pseudo_sign = -1.0 if is_pseudo_covariance else 1.0

        def covariance_integrand(
            integration_domain: np.ndarray | cp.ndarray,
        ) -> np.ndarray | cp.ndarray:
            """The integrand should be of shape N x 6, where N is the number
            of points that need to be evaluated. The final dimension is
            ki_x, ki_y, kj_x, kj_y, ku_x, ku_y"""
            xp = cp.get_array_module(integration_domain)

            # Work out wavevectors
            if is_eight_dimensions:
                ki_x, ki_y, kj_x, kj_y, ku_x, ku_y, kv_x, kv_y = (
                    integration_domain.T
                )
            else:
                ki_x, ki_y, kj_x, kj_y, ku_x, ku_y = integration_domain.T
                kv_x = pseudo_sign * (-ki_x + kj_x) + ku_x
                kv_y = pseudo_sign * (-ki_y + kj_y) + ku_y
            ki_z = ki_z_factor * xp.sqrt(1.0 - ki_x**2 - ki_y**2)
            kj_z = kj_z_factor * xp.sqrt(1.0 - kj_x**2 - kj_y**2)
            ku_z = ku_z_factor * xp.sqrt(1.0 - ku_x**2 - ku_y**2)
            kv_z = kv_z_factor * xp.sqrt(1.0 - kv_x**2 - kv_y**2)

            sinc_factor = special_functions.sinc(
                k * L * (ki_z - kj_z - pseudo_sign * (ku_z - kv_z))
            )
            sec_factor = 1.0 / xp.sqrt(xp.abs(ki_z * kj_z * ku_z * kv_z))

            output = (
                covariance_a_matrix(
                    ki_x,
                    ki_y,
                    ki_z,
                    kj_x,
                    kj_y,
                    kj_z,
                    ku_x,
                    ku_y,
                    ku_z,
                    kv_x,
                    kv_y,
                    kv_z,
                )
                * sinc_factor[:, xp.newaxis]
                * sec_factor[:, xp.newaxis]
            )

            return output

        return covariance_integrand

    def _get_covariance_integration_results_parallelized(
        self,
        covariance_indices: dict[str, dict[str, tuple[int, int, int, int]]],
        integration_method: str = "cubature",
        covariance_cubature_scheme: Any | None = None,
        is_eight_dimensions: bool = True,
    ) -> IntegrationResultList:
        # Main data storage object
        main_result_list = IntegrationResultList()

        master_indices = covariance_indices.get("pp,pp").get("t,t")

        # Multiprocessing parameters
        # None check avoids a bug. If os cannot detect number of cores,
        # just use one
        num_master_indices = len(master_indices)
        num_cores = os.cpu_count()
        num_processes = (
            min(num_master_indices, num_cores) if num_cores is not None else 1
        )
        num_processes = 1
        # Prepare function and arguments for multiprocessing
        parallelised_function = functools.partial(
            self._get_covariance_integration_results_partial,
            covariance_indices=covariance_indices,
            integration_task_config=self.integration_task_config,
            mode_grid=self.mode_grid,
            medium_statistics=self.medium_statistics,
            medium_parameters=self.medium_parameters,
            logger=self.logger,
            integration_method=integration_method,
            covariance_cubature_scheme=covariance_cubature_scheme,
            is_eight_dimensions=is_eight_dimensions,
        )
        partial_master_indices = array_utils.split_list(
            master_indices, num_processes
        )

        # Run function in parallel
        with mp.Pool(processes=num_processes) as pool:
            output = pool.map(parallelised_function, partial_master_indices)

        # Combine results from different processes
        for new_result_list in output:
            main_result_list.merge_result_list(new_result_list)

        return main_result_list

    @staticmethod
    def _get_covariance_integration_results_partial(
        master_indices,
        covariance_indices: dict[str, dict[str, tuple[int, int, int, int]]],
        integration_task_config,
        mode_grid,
        medium_statistics,
        medium_parameters,
        logger,
        integration_method: str = "cubature",
        covariance_cubature_scheme: Any | None = None,
        is_eight_dimensions: bool = True,
    ) -> IntegrationResultList:

        xp = cp if integration_task_config.use_gpu else np

        # Factor common to all covariance integrals
        const_factor = medium_parameters.cov_const_factor
        main_result_list = IntegrationResultList()
        columns_to_keep = [0, 1, 2, 3, 4, 5]

        # Prepare task dictionaries
        covariance_task_dict = {}
        pseudo_covariance_task_dict = {}

        for wave_block in WAVE_BLOCKS:
            wave_block_one, wave_block_two = wave_block.split(",")

            covariance_task_dict[wave_block] = {}
            pseudo_covariance_task_dict[wave_block] = {}

            for block in BLOCKS:
                block_one, block_two = block.split(",")
                block_location = (wave_block, block)

                # Set up covariance and pseudo_covariance tasks separately
                covariance_integrand = get_covariance_integrand(
                    medium_statistics,
                    medium_parameters,
                    wave_block_one,
                    wave_block_two,
                    block_one,
                    block_two,
                    is_eight_dimensions,
                    is_pseudo_covariance=False,
                )
                covariance_task = build_integration_task(
                    integration_method=integration_method,
                    integrand=covariance_integrand,
                    statistic_type="covariance",
                    block_location=block_location,
                    sub_block_locations=[],
                    const_factor=const_factor,
                    use_cupy=integration_task_config.use_gpu,
                    cubature_scheme=covariance_cubature_scheme,
                    use_dirac_density=False,
                    is_eight_dimensions=is_eight_dimensions,
                )
                covariance_task_dict[wave_block][block] = covariance_task

                # Pseudo_covariance
                pseudo_covariance_integrand = get_covariance_integrand(
                    medium_statistics,
                    medium_parameters,
                    wave_block_one,
                    wave_block_two,
                    block_one,
                    block_two,
                    is_eight_dimensions,
                    is_pseudo_covariance=True,
                )
                pseudo_covariance_task = build_integration_task(
                    integration_method=integration_method,
                    integrand=pseudo_covariance_integrand,
                    statistic_type="pseudo_covariance",
                    block_location=block_location,
                    sub_block_locations=[],
                    const_factor=const_factor,
                    use_cupy=integration_task_config.use_gpu,
                    cubature_scheme=covariance_cubature_scheme,
                    use_dirac_density=False,
                    is_eight_dimensions=is_eight_dimensions,
                )
                pseudo_covariance_task_dict[wave_block][
                    block
                ] = pseudo_covariance_task

        # Main loop
        mode_vertices_dict = mode_grid.propagating_modes_vertices_dict
        mean_mode_vertices_dict = (
            mode_grid.propagating_modes_mean_vertices_dict
        )

        # Work out the volume common to the majority of the memory effect
        # type correlations
        repeating_mode_vertices = mode_vertices_dict.get(0)
        cartesian_product = geometry_utils.iterated_cartesian_product(
            [
                repeating_mode_vertices,
                repeating_mode_vertices,
                repeating_mode_vertices,
                repeating_mode_vertices,
            ]
        )
        reduced_intersection = geometry_utils.get_intersection_vertices(
            cartesian_product
        )[:, columns_to_keep]
        reduced_hull = scipy.spatial.ConvexHull(
            reduced_intersection, qhull_options="QJ"
        )
        repeating_volume = reduced_hull.volume
        my_counter = 0
        for indices in logger.progress_bar(master_indices):
            my_counter += 1
            # Work out the template's integration domain
            # This involves the higher dimensional geometry
            i, j, u, v = indices

            centroid = np.concatenate(
                [
                    mean_mode_vertices_dict.get(i),
                    mean_mode_vertices_dict.get(j),
                    mean_mode_vertices_dict.get(u),
                    mean_mode_vertices_dict.get(v),
                ]
            )

            if i == u and j == v:
                # Do it manually for autocorrelations
                # Most of the time it's the repeating thing, but sometimes it's not
                # e.g. when there are edge modes involved

                mode_i_vertices = mode_vertices_dict.get(i)
                mode_j_vertices = mode_vertices_dict.get(j)
                mode_u_vertices = mode_vertices_dict.get(u)
                mode_v_vertices = mode_vertices_dict.get(v)

                # Get the integration domain
                # This part does the geometry with the 8D region being intersected
                # by hyperplanes
                cartesian_product = geometry_utils.iterated_cartesian_product(
                    [
                        mode_i_vertices,
                        mode_j_vertices,
                        mode_u_vertices,
                        mode_v_vertices,
                    ]
                )
                reduced_intersection = (
                    geometry_utils.get_intersection_vertices(
                        cartesian_product
                    )[:, columns_to_keep]
                )
                reduced_hull = scipy.spatial.ConvexHull(
                    reduced_intersection, qhull_options="QJ"
                )
                # centroid = (
                #     xp.mean(cartesian_product, axis=0)
                #     if is_eight_dimensions
                #     else xp.mean(reduced_intersection, axis=0)
                # )
                volume = reduced_hull.volume
            else:
                volume = repeating_volume

            # Set up arrays with derived integration domains
            if integration_method == "cubature":
                centroid_expanded = xp.tile(
                    centroid, (reduced_hull.simplices.shape[0], 1, 1)
                )
                new_simplex_array = xp.concatenate(
                    [
                        xp.asarray(
                            reduced_hull.points[reduced_hull.simplices]
                        ),
                        centroid_expanded,
                    ],
                    axis=1,
                )
            elif integration_method == "midpoint":
                new_midpoint_array = xp.array([centroid])
                new_volume_array = xp.array([volume])

            # Add computed geometric quantities to task dictionaries
            for wave_block in WAVE_BLOCKS:
                wave_block_one, wave_block_two = wave_block.split(",")
                for block in BLOCKS:
                    block_one, block_two = block.split(",")

                    # Check if the mean needs to be calculated for this
                    # particular wave_block, block pair
                    if indices not in covariance_indices.get(
                        wave_block, {}
                    ).get(block, set()):
                        continue

                    # Add domain to integral task
                    if integration_method == "cubature":
                        old_stack_length = len(
                            covariance_task_dict[wave_block][
                                block
                            ].simplex_array
                        )
                        new_stack_length = old_stack_length + len(
                            new_simplex_array
                        )

                        covariance_task_dict[wave_block][
                            block
                        ].simplex_array = xp.vstack(
                            (
                                covariance_task_dict[wave_block][
                                    block
                                ].simplex_array,
                                new_simplex_array,
                            )
                        )
                        pseudo_covariance_task_dict[wave_block][
                            block
                        ].simplex_array = xp.vstack(
                            (
                                pseudo_covariance_task_dict[wave_block][
                                    block
                                ].simplex_array,
                                new_simplex_array,
                            )
                        )

                    elif integration_method == "midpoint":
                        old_stack_length = len(
                            covariance_task_dict[wave_block][
                                block
                            ].midpoint_array
                        )
                        new_stack_length = old_stack_length + len(
                            new_midpoint_array
                        )

                        covariance_task_dict[wave_block][
                            block
                        ].midpoint_array = xp.vstack(
                            (
                                covariance_task_dict[wave_block][
                                    block
                                ].midpoint_array,
                                new_midpoint_array,
                            )
                        )
                        covariance_task_dict[wave_block][
                            block
                        ].volume_array = xp.vstack(
                            (
                                covariance_task_dict[wave_block][
                                    block
                                ].volume_array,
                                new_volume_array,
                            )
                        )

                        pseudo_covariance_task_dict[wave_block][
                            block
                        ].midpoint_array = xp.vstack(
                            (
                                covariance_task_dict[wave_block][
                                    block
                                ].midpoint_array,
                                new_midpoint_array,
                            )
                        )
                        pseudo_covariance_task_dict[wave_block][
                            block
                        ].volume_array = xp.vstack(
                            (
                                covariance_task_dict[wave_block][
                                    block
                                ].volume_array,
                                new_volume_array,
                            )
                        )

                    # Add sub block locations.
                    # Remember that u and v are negated for pseudo_covariance
                    # for reciprocal mode grids
                    new_slice = slice(old_stack_length, new_stack_length)

                    new_covariance_indices = indices
                    new_covariance_sub_block_location = (
                        new_slice,
                        new_covariance_indices,
                    )
                    covariance_task_dict[wave_block][
                        block
                    ].sub_block_locations.append(
                        new_covariance_sub_block_location
                    )

                    new_pseudo_covariance_indices = (i, j, -u, -v)
                    new_pseudo_covariance_sub_block_location = (
                        new_slice,
                        new_pseudo_covariance_indices,
                    )
                    pseudo_covariance_task_dict[wave_block][
                        block
                    ].sub_block_locations.append(
                        new_pseudo_covariance_sub_block_location
                    )

                    # Execute tasks if RAM usage is getting too high.
                    # Re-initialize relevant integration task
                    current_ram_usage = system_utils.get_current_ram_usage()
                    # if current_ram_usage > integration_task_config.ram_limit:
                    if True:
                        new_covariance_result = covariance_task_dict[
                            wave_block
                        ][block].execute_task()
                        main_result_list.append_result(new_covariance_result)

                        block_location = (wave_block, block)

                        covariance_integrand = get_covariance_integrand(
                            medium_statistics,
                            medium_parameters,
                            wave_block_one,
                            wave_block_two,
                            block_one,
                            block_two,
                            is_eight_dimensions,
                            is_pseudo_covariance=False,
                        )

                        covariance_task = build_integration_task(
                            integration_method=integration_method,
                            integrand=covariance_integrand,
                            statistic_type="covariance",
                            block_location=block_location,
                            sub_block_locations=[],
                            const_factor=const_factor,
                            use_cupy=integration_task_config.use_gpu,
                            cubature_scheme=covariance_cubature_scheme,
                            use_dirac_density=False,
                            is_eight_dimensions=is_eight_dimensions,
                        )
                        covariance_task_dict[wave_block][
                            block
                        ] = covariance_task

                        new_pseudo_covariance_result = (
                            pseudo_covariance_task_dict[wave_block][
                                block
                            ].execute_task()
                        )
                        main_result_list.append_result(
                            new_pseudo_covariance_result
                        )

                        block_location = (wave_block, block)

                        pseudo_covariance_integrand = get_covariance_integrand(
                            medium_statistics,
                            medium_parameters,
                            wave_block_one,
                            wave_block_two,
                            block_one,
                            block_two,
                            is_eight_dimensions,
                            is_pseudo_covariance=True,
                        )

                        pseudo_covariance_task = build_integration_task(
                            integration_method=integration_method,
                            integrand=pseudo_covariance_integrand,
                            statistic_type="pseudo_covariance",
                            block_location=block_location,
                            sub_block_locations=[],
                            const_factor=const_factor,
                            use_cupy=integration_task_config.use_gpu,
                            cubature_scheme=covariance_cubature_scheme,
                            use_dirac_density=False,
                            is_eight_dimensions=is_eight_dimensions,
                        )
                        pseudo_covariance_task_dict[wave_block][
                            block
                        ] = pseudo_covariance_task

        # Execute remaining tasks
        for wave_block in WAVE_BLOCKS:
            for block in BLOCKS:
                new_covariance_result = covariance_task_dict[wave_block][
                    block
                ].execute_task()
                main_result_list.append_result(new_covariance_result)

                new_pseudo_covariance_result = pseudo_covariance_task_dict[
                    wave_block
                ][block].execute_task()
                main_result_list.append_result(new_pseudo_covariance_result)

        return main_result_list

    def _get_covariance_integration_results(
        self,
        covariance_indices: dict[str, dict[str, tuple[int, int, int, int]]],
        integration_method: str = "cubature",
        covariance_cubature_scheme: Any | None = None,
        is_eight_dimensions: bool = True,
    ) -> IntegrationResultList:
        """Main method for computing covariance results"""
        xp = cp if self.integration_task_config.use_gpu else np

        # Factor common to all covariance integrals
        const_factor = self.medium_parameters.cov_const_factor
        main_result_list = IntegrationResultList()
        columns_to_keep = [0, 1, 2, 3, 4, 5]

        # Prepare task dictionaries
        covariance_task_dict = {}
        pseudo_covariance_task_dict = {}

        for wave_block in WAVE_BLOCKS:
            wave_block_one, wave_block_two = wave_block.split(",")

            covariance_task_dict[wave_block] = {}
            pseudo_covariance_task_dict[wave_block] = {}

            for block in BLOCKS:
                block_one, block_two = block.split(",")
                block_location = (wave_block, block)

                # Set up covariance and pseudo_covariance tasks separately
                covariance_integrand = self._get_covariance_integrand(
                    wave_block_one,
                    wave_block_two,
                    block_one,
                    block_two,
                    is_eight_dimensions,
                    is_pseudo_covariance=False,
                )
                covariance_task = build_integration_task(
                    integration_method=integration_method,
                    integrand=covariance_integrand,
                    statistic_type="covariance",
                    block_location=block_location,
                    sub_block_locations=[],
                    const_factor=const_factor,
                    use_cupy=self.integration_task_config.use_gpu,
                    cubature_scheme=covariance_cubature_scheme,
                    use_dirac_density=False,
                    is_eight_dimensions=is_eight_dimensions,
                )
                covariance_task_dict[wave_block][block] = covariance_task

                # Pseudo_covariance
                pseudo_covariance_integrand = self._get_covariance_integrand(
                    wave_block_one,
                    wave_block_two,
                    block_one,
                    block_two,
                    is_eight_dimensions,
                    is_pseudo_covariance=True,
                )
                pseudo_covariance_task = build_integration_task(
                    integration_method=integration_method,
                    integrand=pseudo_covariance_integrand,
                    statistic_type="pseudo_covariance",
                    block_location=block_location,
                    sub_block_locations=[],
                    const_factor=const_factor,
                    use_cupy=self.integration_task_config.use_gpu,
                    cubature_scheme=covariance_cubature_scheme,
                    use_dirac_density=False,
                    is_eight_dimensions=is_eight_dimensions,
                )
                pseudo_covariance_task_dict[wave_block][
                    block
                ] = pseudo_covariance_task

        # Main loop
        master_indices = covariance_indices.get("pp,pp").get("t,t")
        mode_vertices_dict = self.mode_grid.propagating_modes_vertices_dict
        mean_mode_vertices_dict = (
            self.mode_grid.propagating_modes_mean_vertices_dict
        )

        # Work out the volume common to the majority of the memory effect
        # type correlations
        repeating_mode_vertices = mode_vertices_dict.get(0)
        cartesian_product = geometry_utils.iterated_cartesian_product(
            [
                repeating_mode_vertices,
                repeating_mode_vertices,
                repeating_mode_vertices,
                repeating_mode_vertices,
            ]
        )
        reduced_intersection = geometry_utils.get_intersection_vertices(
            cartesian_product
        )[:, columns_to_keep]
        reduced_hull = scipy.spatial.ConvexHull(
            reduced_intersection, qhull_options="QJ"
        )
        repeating_volume = reduced_hull.volume

        for indices in self.logger.progress_bar(master_indices):
            # Work out the template's integration domain
            # This involves the higher dimensional geometry
            i, j, u, v = indices

            centroid = np.concatenate(
                [
                    mean_mode_vertices_dict.get(i),
                    mean_mode_vertices_dict.get(j),
                    mean_mode_vertices_dict.get(u),
                    mean_mode_vertices_dict.get(v),
                ]
            )

            if i == u and j == v:
                # Do it manually for autocorrelations
                # Most of the time it's the repeating thing, but sometimes it's not
                # e.g. when there are edge modes involved

                mode_i_vertices = mode_vertices_dict.get(i)
                mode_j_vertices = mode_vertices_dict.get(j)
                mode_u_vertices = mode_vertices_dict.get(u)
                mode_v_vertices = mode_vertices_dict.get(v)

                # Get the integration domain
                # This part does the geometry with the 8D region being intersected
                # by hyperplanes
                cartesian_product = geometry_utils.iterated_cartesian_product(
                    [
                        mode_i_vertices,
                        mode_j_vertices,
                        mode_u_vertices,
                        mode_v_vertices,
                    ]
                )
                reduced_intersection = (
                    geometry_utils.get_intersection_vertices(
                        cartesian_product
                    )[:, columns_to_keep]
                )
                reduced_hull = scipy.spatial.ConvexHull(
                    reduced_intersection, qhull_options="QJ"
                )
                # centroid = (
                #     xp.mean(cartesian_product, axis=0)
                #     if is_eight_dimensions
                #     else xp.mean(reduced_intersection, axis=0)
                # )
                volume = reduced_hull.volume
            else:
                volume = repeating_volume

            # Set up arrays with derived integration domains
            if integration_method == "cubature":
                centroid_expanded = xp.tile(
                    centroid, (reduced_hull.simplices.shape[0], 1, 1)
                )
                new_simplex_array = xp.concatenate(
                    [
                        xp.asarray(
                            reduced_hull.points[reduced_hull.simplices]
                        ),
                        centroid_expanded,
                    ],
                    axis=1,
                )
            elif integration_method == "midpoint":
                new_midpoint_array = xp.array([centroid])
                new_volume_array = xp.array([volume])

            # Add computed geometric quantities to task dictionaries
            for wave_block in WAVE_BLOCKS:
                wave_block_one, wave_block_two = wave_block.split(",")
                for block in BLOCKS:
                    block_one, block_two = block.split(",")

                    # Check if the mean needs to be calculated for this
                    # particular wave_block, block pair
                    if indices not in covariance_indices.get(
                        wave_block, {}
                    ).get(block, set()):
                        continue

                    # Add domain to integral task
                    if integration_method == "cubature":
                        old_stack_length = len(
                            covariance_task_dict[wave_block][
                                block
                            ].simplex_array
                        )
                        new_stack_length = old_stack_length + len(
                            new_simplex_array
                        )

                        covariance_task_dict[wave_block][
                            block
                        ].simplex_array = xp.vstack(
                            (
                                covariance_task_dict[wave_block][
                                    block
                                ].simplex_array,
                                new_simplex_array,
                            )
                        )
                        pseudo_covariance_task_dict[wave_block][
                            block
                        ].simplex_array = xp.vstack(
                            (
                                pseudo_covariance_task_dict[wave_block][
                                    block
                                ].simplex_array,
                                new_simplex_array,
                            )
                        )

                    elif integration_method == "midpoint":
                        old_stack_length = len(
                            covariance_task_dict[wave_block][
                                block
                            ].midpoint_array
                        )
                        new_stack_length = old_stack_length + len(
                            new_midpoint_array
                        )

                        covariance_task_dict[wave_block][
                            block
                        ].midpoint_array = xp.vstack(
                            (
                                covariance_task_dict[wave_block][
                                    block
                                ].midpoint_array,
                                new_midpoint_array,
                            )
                        )
                        covariance_task_dict[wave_block][
                            block
                        ].volume_array = xp.vstack(
                            (
                                covariance_task_dict[wave_block][
                                    block
                                ].volume_array,
                                new_volume_array,
                            )
                        )

                        pseudo_covariance_task_dict[wave_block][
                            block
                        ].midpoint_array = xp.vstack(
                            (
                                covariance_task_dict[wave_block][
                                    block
                                ].midpoint_array,
                                new_midpoint_array,
                            )
                        )
                        pseudo_covariance_task_dict[wave_block][
                            block
                        ].volume_array = xp.vstack(
                            (
                                covariance_task_dict[wave_block][
                                    block
                                ].volume_array,
                                new_volume_array,
                            )
                        )

                    # Add sub block locations.
                    # Remember that u and v are negated for pseudo_covariance
                    # for reciprocal mode grids
                    new_slice = slice(old_stack_length, new_stack_length)

                    new_covariance_indices = indices
                    new_covariance_sub_block_location = (
                        new_slice,
                        new_covariance_indices,
                    )
                    covariance_task_dict[wave_block][
                        block
                    ].sub_block_locations.append(
                        new_covariance_sub_block_location
                    )

                    new_pseudo_covariance_indices = (i, j, -u, -v)
                    new_pseudo_covariance_sub_block_location = (
                        new_slice,
                        new_pseudo_covariance_indices,
                    )
                    pseudo_covariance_task_dict[wave_block][
                        block
                    ].sub_block_locations.append(
                        new_pseudo_covariance_sub_block_location
                    )

                    # Execute tasks if RAM usage is getting too high.
                    # Re-initialize relevant integration task
                    current_ram_usage = (
                        self.integration_task_config.get_current_ram_usage()
                    )
                    if (
                        current_ram_usage
                        > self.integration_task_config.ram_limit
                    ):
                        new_covariance_result = covariance_task_dict[
                            wave_block
                        ][block].execute_task()
                        main_result_list.append_result(new_covariance_result)

                        block_location = (wave_block, block)

                        covariance_integrand = self._get_covariance_integrand(
                            wave_block_one,
                            wave_block_two,
                            block_one,
                            block_two,
                            is_eight_dimensions,
                            is_pseudo_covariance=False,
                        )

                        covariance_task = build_integration_task(
                            integration_method=integration_method,
                            integrand=covariance_integrand,
                            statistic_type="covariance",
                            block_location=block_location,
                            sub_block_locations=[],
                            const_factor=const_factor,
                            use_cupy=self.integration_task_config.use_gpu,
                            cubature_scheme=covariance_cubature_scheme,
                            use_dirac_density=False,
                            is_eight_dimensions=is_eight_dimensions,
                        )
                        covariance_task_dict[wave_block][
                            block
                        ] = covariance_task

                        new_pseudo_covariance_result = (
                            pseudo_covariance_task_dict[wave_block][
                                block
                            ].execute_task()
                        )
                        main_result_list.append_result(
                            new_pseudo_covariance_result
                        )

                        block_location = (wave_block, block)

                        pseudo_covariance_integrand = (
                            self._get_covariance_integrand(
                                wave_block_one,
                                wave_block_two,
                                block_one,
                                block_two,
                                is_eight_dimensions,
                                is_pseudo_covariance=True,
                            )
                        )

                        pseudo_covariance_task = build_integration_task(
                            integration_method=integration_method,
                            integrand=pseudo_covariance_integrand,
                            statistic_type="pseudo_covariance",
                            block_location=block_location,
                            sub_block_locations=[],
                            const_factor=const_factor,
                            use_cupy=self.integration_task_config.use_gpu,
                            cubature_scheme=covariance_cubature_scheme,
                            use_dirac_density=False,
                            is_eight_dimensions=is_eight_dimensions,
                        )
                        pseudo_covariance_task_dict[wave_block][
                            block
                        ] = pseudo_covariance_task

        # Execute remaining tasks
        for wave_block in WAVE_BLOCKS:
            for block in BLOCKS:
                new_covariance_result = covariance_task_dict[wave_block][
                    block
                ].execute_task()
                main_result_list.append_result(new_covariance_result)

                new_pseudo_covariance_result = pseudo_covariance_task_dict[
                    wave_block
                ][block].execute_task()
                main_result_list.append_result(new_pseudo_covariance_result)

        return main_result_list


def get_covariance_integrand(
    medium_statistics,
    medium_parameters,
    wave_block_one: str,
    wave_block_two: str,
    block_one: str,
    block_two: str,
    is_eight_dimensions: bool = False,
    is_pseudo_covariance: bool = False,
) -> MathematicalFunction:
    """Get the integrand for covariance calculations. Note that this does
    not include mode weights. We choose to add those in at the end."""

    covariance_a_matrix = (
        medium_statistics.get_covariance_a_matrix()
        if not is_pseudo_covariance
        else medium_statistics.get_pseudo_covariance_a_matrix()
    )
    k = medium_parameters.k
    L = medium_parameters.L

    ki_z_factor = -1.0 if block_one in {"r2", "t2"} else 1.0
    kj_z_factor = -1.0 if block_one in {"r", "t2"} else 1.0
    ku_z_factor = -1.0 if block_two in {"r2", "t2"} else 1.0
    kv_z_factor = -1.0 if block_two in {"r", "t2"} else 1.0
    pseudo_sign = -1.0 if is_pseudo_covariance else 1.0

    def covariance_integrand(
        integration_domain: np.ndarray | cp.ndarray,
    ) -> np.ndarray | cp.ndarray:
        """The integrand should be of shape N x 6, where N is the number
        of points that need to be evaluated. The final dimension is
        ki_x, ki_y, kj_x, kj_y, ku_x, ku_y"""
        xp = cp.get_array_module(integration_domain)

        # Work out wavevectors
        if is_eight_dimensions:
            ki_x, ki_y, kj_x, kj_y, ku_x, ku_y, kv_x, kv_y = (
                integration_domain.T
            )
        else:
            ki_x, ki_y, kj_x, kj_y, ku_x, ku_y = integration_domain.T
            kv_x = pseudo_sign * (-ki_x + kj_x) + ku_x
            kv_y = pseudo_sign * (-ki_y + kj_y) + ku_y
        ki_z = ki_z_factor * xp.sqrt(1.0 - ki_x**2 - ki_y**2)
        kj_z = kj_z_factor * xp.sqrt(1.0 - kj_x**2 - kj_y**2)
        ku_z = ku_z_factor * xp.sqrt(1.0 - ku_x**2 - ku_y**2)
        kv_z = kv_z_factor * xp.sqrt(1.0 - kv_x**2 - kv_y**2)

        sinc_factor = special_functions.sinc(
            k * L * (ki_z - kj_z - pseudo_sign * (ku_z - kv_z))
        )
        sec_factor = 1.0 / xp.sqrt(xp.abs(ki_z * kj_z * ku_z * kv_z))

        output = (
            covariance_a_matrix(
                ki_x,
                ki_y,
                ki_z,
                kj_x,
                kj_y,
                kj_z,
                ku_x,
                ku_y,
                ku_z,
                kv_x,
                kv_y,
                kv_z,
            )
            * sinc_factor[:, xp.newaxis]
            * sec_factor[:, xp.newaxis]
        )

        return output

    return covariance_integrand



    # ------------------------------
    # GENERATORS
    # -----------------------------

    # def _get_covariance_integrand(
    #     self,
    #     wave_block_one: str,
    #     wave_block_two: str,
    #     block_one: str,
    #     block_two: str,
    # ) -> MathematicalFunction:
    #     covariance_a_matrix = self.medium_statistics.get_covariance_a_matrix()
    #     k = self.medium_parameters.k
    #     L = self.medium_parameters.L

    #     ki_z_factor = -1.0 if block_one in {"r2", "t2"} else 1.0
    #     kj_z_factor = -1.0 if block_one in {"r", "t2"} else 1.0
    #     ku_z_factor = -1.0 if block_two in {"r2", "t2"} else 1.0
    #     kv_z_factor = -1.0 if block_two in {"r", "t2"} else 1.0

    #     def covariance_integrand(
    #         integration_domain: np.ndarray | cp.ndarray,
    #     ) -> np.ndarray | cp.ndarray:
    #         """The integrand should be of shape N x 6, where N is the number
    #         of points that need to be evaluated. The final dimension is
    #         ki_x, ki_y, kj_x, kj_y, ku_x, ku_y"""
    #         xp = cp.get_array_module(integration_domain)
    #         ki_x, ki_y, kj_x, kj_y, ku_x, ku_y = integration_domain.T

    #         kv_x = -ki_x + kj_x + ku_x
    #         kv_y = -ki_y + kj_y + ku_y

    #         ki_z = ki_z_factor * xp.sqrt(1.0 - ki_x**2 - ki_y**2)
    #         kj_z = kj_z_factor * xp.sqrt(1.0 - kj_x**2 - kj_y**2)
    #         ku_z = ku_z_factor * xp.sqrt(1.0 - ku_x**2 - ku_y**2)
    #         kv_z = kv_z_factor * xp.sqrt(1.0 - kv_x**2 - kv_y**2)

    #         sinc_factor = special_functions.sinc(
    #             k * L * (ki_z - kj_z - ku_z + kv_z)
    #         )
    #         sec_factor = 1.0 / xp.sqrt(xp.abs(ki_z * kj_z * ku_z * kv_z))

    #         output = (
    #             covariance_a_matrix(
    #                 ki_x,
    #                 ki_y,
    #                 ki_z,
    #                 kj_x,
    #                 kj_y,
    #                 kj_z,
    #                 ku_x,
    #                 ku_y,
    #                 ku_z,
    #                 kv_x,
    #                 kv_y,
    #                 kv_z,
    #             )
    #             * sinc_factor[:, xp.newaxis]
    #             * sec_factor[:, xp.newaxis]
    #         )

    #         return output

    #     return covariance_integrand

    # def _get_covariance_integrand_eight(
    #     self,
    #     wave_block_one: str,
    #     wave_block_two: str,
    #     block_one: str,
    #     block_two: str,
    # ) -> MathematicalFunction:
    #     covariance_a_matrix = self.medium_statistics.get_covariance_a_matrix()
    #     k = self.medium_parameters.k
    #     L = self.medium_parameters.L

    #     ki_z_factor = -1.0 if block_one in {"r2", "t2"} else 1.0
    #     kj_z_factor = -1.0 if block_one in {"r", "t2"} else 1.0
    #     ku_z_factor = -1.0 if block_two in {"r2", "t2"} else 1.0
    #     kv_z_factor = -1.0 if block_two in {"r", "t2"} else 1.0

    #     def covariance_integrand(
    #         integration_domain: np.ndarray | cp.ndarray,
    #     ) -> np.ndarray | cp.ndarray:
    #         """The integrand should be of shape N x 8, where N is the number
    #         of points that need to be evaluated. The final dimension is
    #         ki_x, ki_y, kj_x, kj_y, ku_x, ku_y"""
    #         xp = cp.get_array_module(integration_domain)
    #         ki_x, ki_y, kj_x, kj_y, ku_x, ku_y, kv_x, kv_y = (
    #             integration_domain.T
    #         )

    #         ki_z = ki_z_factor * xp.sqrt(1.0 - ki_x**2 - ki_y**2)
    #         kj_z = kj_z_factor * xp.sqrt(1.0 - kj_x**2 - kj_y**2)
    #         ku_z = ku_z_factor * xp.sqrt(1.0 - ku_x**2 - ku_y**2)
    #         kv_z = kv_z_factor * xp.sqrt(1.0 - kv_x**2 - kv_y**2)

    #         sinc_factor = special_functions.sinc(
    #             k * L * (ki_z - kj_z - ku_z + kv_z)
    #         )
    #         sec_factor = 1.0 / xp.sqrt(xp.abs(ki_z * kj_z * ku_z * kv_z))

    #         output = (
    #             covariance_a_matrix(
    #                 ki_x,
    #                 ki_y,
    #                 ki_z,
    #                 kj_x,
    #                 kj_y,
    #                 kj_z,
    #                 ku_x,
    #                 ku_y,
    #                 ku_z,
    #                 kv_x,
    #                 kv_y,
    #                 kv_z,
    #             )
    #             * sinc_factor[:, xp.newaxis]
    #             * sec_factor[:, xp.newaxis]
    #         )

    #         return output

    #     return covariance_integrand

    # def _get_pseudo_covariance_integrand(
    #     self,
    #     wave_block_one: str,
    #     wave_block_two: str,
    #     block_one: str,
    #     block_two: str,
    # ) -> MathematicalFunction:
    #     pseudo_covariance_a_matrix = (
    #         self.medium_statistics.get_pseudo_covariance_a_matrix()
    #     )
    #     k = self.medium_parameters.k
    #     L = self.medium_parameters.L

    #     ki_z_factor = -1.0 if block_one in {"r2", "t2"} else 1.0
    #     kj_z_factor = -1.0 if block_one in {"r", "t2"} else 1.0
    #     ku_z_factor = -1.0 if block_two in {"r2", "t2"} else 1.0
    #     kv_z_factor = -1.0 if block_two in {"r", "t2"} else 1.0

    #     def pseudo_covariance_integrand(
    #         integration_domain: np.ndarray | cp.ndarray,
    #     ) -> np.ndarray | cp.ndarray:
    #         """The integrand should be of shape N x 6, where N is the number
    #         of points that need to be evaluated. The final dimension is
    #         ki_x, ki_y, kj_x, kj_y, ku_x, ku_y"""
    #         xp = cp.get_array_module(integration_domain)
    #         ki_x, ki_y, kj_x, kj_y, ku_x, ku_y = integration_domain.T

    #         kv_x = -ki_x + kj_x + ku_x
    #         kv_y = -ki_y + kj_y + ku_y

    #         ki_z = ki_z_factor * xp.sqrt(1.0 - ki_x**2 - ki_y**2)
    #         kj_z = kj_z_factor * xp.sqrt(1.0 - kj_x**2 - kj_y**2)
    #         ku_z = ku_z_factor * xp.sqrt(1.0 - ku_x**2 - ku_y**2)
    #         kv_z = kv_z_factor * xp.sqrt(1.0 - kv_x**2 - kv_y**2)

    #         sinc_factor = special_functions.sinc(
    #             k * L * (ki_z - kj_z + ku_z - kv_z)
    #         )
    #         sec_factor = 1.0 / xp.sqrt(xp.abs(ki_z * kj_z * ku_z * kv_z))

    #         output = (
    #             pseudo_covariance_a_matrix(
    #                 ki_x,
    #                 ki_y,
    #                 ki_z,
    #                 kj_x,
    #                 kj_y,
    #                 kj_z,
    #                 ku_x,
    #                 ku_y,
    #                 ku_z,
    #                 kv_x,
    #                 kv_y,
    #                 kv_z,
    #             )
    #             * sinc_factor[:, xp.newaxis]
    #             * sec_factor[:, xp.newaxis]
    #         )

    #         return output

    #     return pseudo_covariance_integrand

    # def _get_covariance_integrand_symmetric(
    #     self,
    #     wave_block_one: str,
    #     wave_block_two: str,
    #     block_one: str,
    #     block_two: str,
    # ) -> MathematicalFunction:
    #     covariance_a_matrix = self.medium_statistics.get_covariance_a_matrix()
    #     k = self.medium_parameters.k
    #     L = self.medium_parameters.L

    #     ki_z_factor = -1.0 if block_one in {"r2", "t2"} else 1.0
    #     kj_z_factor = -1.0 if block_one in {"r", "t2"} else 1.0
    #     ku_z_factor = -1.0 if block_two in {"r2", "t2"} else 1.0
    #     kv_z_factor = -1.0 if block_two in {"r", "t2"} else 1.0

    #     def covariance_integrand(
    #         integration_domain: np.ndarray | cp.ndarray,
    #     ) -> np.ndarray | cp.ndarray:
    #         """The integrand should be of shape N x 6, where N is the number
    #         of points that need to be evaluated. The final dimension is
    #         ki_x, ki_y, kj_x, kj_y, ku_x, ku_y"""
    #         xp = cp.get_array_module(integration_domain)
    #         ki_x, ki_y, kj_y, ku_x, ku_y, kv_x = integration_domain.T

    #         kj_x = kv_x + ki_x - ku_x
    #         kv_y = -ki_y + kj_y + ku_y

    #         ki_z = ki_z_factor * xp.sqrt(1 - ki_x**2 - ki_y**2)
    #         kj_z = kj_z_factor * xp.sqrt(1 - kj_x**2 - kj_y**2)
    #         ku_z = ku_z_factor * xp.sqrt(1 - ku_x**2 - ku_y**2)
    #         kv_z = kv_z_factor * xp.sqrt(1 - kv_x**2 - kv_y**2)

    #         sinc_factor = special_functions.sinc(
    #             k * L * (ki_z - kj_z - ku_z + kv_z)
    #         )
    #         sec_factor = 1.0 / xp.abs(np.sqrt(ki_z * kj_z * ku_z * kv_z))

    #         output = (
    #             covariance_a_matrix(
    #                 ki_x,
    #                 ki_y,
    #                 ki_z,
    #                 kj_x,
    #                 kj_y,
    #                 kj_z,
    #                 ku_x,
    #                 ku_y,
    #                 ku_z,
    #                 kv_x,
    #                 kv_y,
    #                 kv_z,
    #             )
    #             * sinc_factor[:, xp.newaxis]
    #             * sec_factor[:, xp.newaxis]
    #         )

    #         return output

    #     return covariance_integrand

    # def _get_covariance_integrand_dirac_density(
    #     self,
    #     wave_block_one: str,
    #     wave_block_two: str,
    #     block_one: str,
    #     block_two: str,
    #     kj_centroid: np.ndarray | cp.ndarray,
    #     kv_centroid: np.ndarray | cp.ndarray,
    # ) -> MathematicalFunction:
    #     covariance_a_matrix = self.medium_statistics.get_covariance_a_matrix()
    #     k = self.medium_parameters.k
    #     L = self.medium_parameters.L

    #     ki_z_factor = -1.0 if block_one in {"r2", "t2"} else 1.0
    #     kj_z_factor = -1.0 if block_one in {"r", "t2"} else 1.0
    #     ku_z_factor = -1.0 if block_two in {"r2", "t2"} else 1.0
    #     kv_z_factor = -1.0 if block_two in {"r", "t2"} else 1.0

    #     def covariance_integrand(
    #         integration_domain: np.ndarray | cp.ndarray,
    #     ) -> np.ndarray | cp.ndarray:
    #         """The integrand should be of shape N x 2, where N is the number
    #         of points that need to be evaluated. The final dimension is
    #         ki_x, ki_y"""
    #         xp = cp.get_array_module(integration_domain)
    #         ki_x, ki_y = integration_domain.T
    #         num_entries = len(ki_x)

    #         kj_x = xp.repeat(kj_centroid[0], num_entries)
    #         kj_y = xp.repeat(kj_centroid[1], num_entries)
    #         kv_x = xp.repeat(kv_centroid[0], num_entries)
    #         kv_y = xp.repeat(kv_centroid[1], num_entries)

    #         ku_x = ki_x - kj_x + kv_x
    #         ku_y = ki_y - kj_y + kv_y

    #         ki_z = ki_z_factor * xp.sqrt(1 - ki_x**2 - ki_y**2)
    #         kj_z = kj_z_factor * xp.sqrt(1 - kj_x**2 - kj_y**2)
    #         ku_z = ku_z_factor * xp.sqrt(1 - ku_x**2 - ku_y**2)
    #         kv_z = kv_z_factor * xp.sqrt(1 - kv_x**2 - kv_y**2)

    #         sinc_factor = special_functions.sinc(
    #             k * L * (ki_z - kj_z - ku_z + kv_z)
    #         )
    #         sec_factor = 1.0 / xp.abs(np.sqrt(ki_z * kj_z * ku_z * kv_z))

    #         output = (
    #             covariance_a_matrix(
    #                 ki_x,
    #                 ki_y,
    #                 ki_z,
    #                 kj_x,
    #                 kj_y,
    #                 kj_z,
    #                 ku_x,
    #                 ku_y,
    #                 ku_z,
    #                 kv_x,
    #                 kv_y,
    #                 kv_z,
    #             )
    #             * sinc_factor[:, xp.newaxis]
    #             * sec_factor[:, xp.newaxis]
    #         )

    #         return output

    #     return covariance_integrand

    # def _get_covariance_integration_results_six_dimensions(
    #     self,
    #     covariance_indices: dict[str, dict[str, tuple[int, int, int, int]]],
    #     integration_method: str = "cubature",
    #     covariance_cubature_scheme: Any | None = None,
    # ) -> IntegrationResultList:
    #     """Main method for computing covariance results based on
    #     six dimensional integrals"""
    #     xp = cp if self.integration_task_config.use_gpu else np

    #     # Factor common to all covariance integrals
    #     const_factor = self.medium_parameters.cov_const_factor
    #     main_result_list = IntegrationResultList()
    #     columns_to_keep = [0, 1, 2, 3, 4, 5]

    #     # Prepare task dictionaries
    #     covariance_task_dict = {}
    #     pseudo_covariance_task_dict = {}

    #     for wave_block in WAVE_BLOCKS:
    #         wave_block_one, wave_block_two = wave_block.split(",")

    #         covariance_task_dict[wave_block] = {}
    #         pseudo_covariance_task_dict[wave_block] = {}

    #         for block in BLOCKS:
    #             block_one, block_two = block.split(",")
    #             block_location = (wave_block, block)

    #             # Set up covariance and pseudo_covariance separately
    #             covariance_integrand = self._get_covariance_integrand(
    #                 wave_block_one, wave_block_two, block_one, block_two
    #             )
    #             covariance_task = build_integration_task(
    #                 integration_method=integration_method,
    #                 integrand=covariance_integrand,
    #                 statistic_type="covariance",
    #                 block_location=block_location,
    #                 sub_block_locations=[],
    #                 const_factor=const_factor,
    #                 use_cupy=self.integration_task_config.use_gpu,
    #                 cubature_scheme=covariance_cubature_scheme,
    #                 use_dirac_density=False,
    #             )
    #             covariance_task_dict[wave_block][block] = covariance_task

    #             # Pseudo_covariance
    #             pseudo_covariance_integrand = (
    #                 self._get_pseudo_covariance_integrand(
    #                     wave_block_one, wave_block_two, block_one, block_two
    #                 )
    #             )
    #             pseudo_covariance_task = build_integration_task(
    #                 integration_method=integration_method,
    #                 integrand=pseudo_covariance_integrand,
    #                 statistic_type="pseudo_covariance",
    #                 block_location=block_location,
    #                 sub_block_locations=[],
    #                 const_factor=const_factor,
    #                 use_cupy=self.integration_task_config.use_gpu,
    #                 cubature_scheme=covariance_cubature_scheme,
    #                 use_dirac_density=False,
    #             )
    #             pseudo_covariance_task_dict[wave_block][
    #                 block
    #             ] = pseudo_covariance_task

    #     # Main loop
    #     master_indices = covariance_indices.get("pp,pp").get("t,t")
    #     mode_vertices_dict = self.mode_grid.propagating_modes_vertices_dict

    #     for indices in self.logger.progress_bar(master_indices):
    #         # Work out the template's integration domain
    #         # This involves the higher dimensional geometry
    #         i, j, u, v = indices
    #         mode_i_vertices = mode_vertices_dict.get(i)
    #         mode_j_vertices = mode_vertices_dict.get(j)
    #         mode_u_vertices = mode_vertices_dict.get(u)
    #         mode_v_vertices = mode_vertices_dict.get(v)

    #         # Get the integration domain
    #         cartesian_product = geometry_utils.iterated_cartesian_product(
    #             [
    #                 mode_i_vertices,
    #                 mode_j_vertices,
    #                 mode_u_vertices,
    #                 mode_v_vertices,
    #             ]
    #         )
    #         reduced_intersection = geometry_utils.get_intersection_vertices(
    #             cartesian_product
    #         )[:, columns_to_keep]
    #         reduced_hull = scipy.spatial.ConvexHull(
    #             reduced_intersection, qhull_options="QJ"
    #         )
    #         centroid = xp.mean(reduced_intersection, axis=0)
    #         volume = reduced_hull.volume

    #         if integration_method == "cubature":
    #             centroid_expanded = xp.tile(
    #                 centroid, (reduced_hull.simplices.shape[0], 1, 1)
    #             )
    #             new_simplex_array = xp.concatenate(
    #                 [
    #                     xp.asarray(
    #                         reduced_hull.points[reduced_hull.simplices]
    #                     ),
    #                     centroid_expanded,
    #                 ],
    #                 axis=1,
    #             )
    #         elif integration_method == "midpoint":
    #             new_midpoint_array = xp.array([centroid])
    #             new_volume_array = xp.array([volume])

    #         # Add computed geometric quantities to task dictionaries
    #         for wave_block in WAVE_BLOCKS:
    #             wave_block_one, wave_block_two = wave_block.split(",")
    #             for block in BLOCKS:
    #                 block_one, block_two = block.split(",")

    #                 # Check if the mean needs to be calculated for this
    #                 # particular wave_block, block pair
    #                 if indices not in covariance_indices.get(
    #                     wave_block, {}
    #                 ).get(block, set()):
    #                     continue

    #                 # Add domain to integral task
    #                 if integration_method == "cubature":
    #                     old_stack_length = len(
    #                         covariance_task_dict[wave_block][
    #                             block
    #                         ].simplex_array
    #                     )
    #                     new_stack_length = old_stack_length + len(
    #                         new_simplex_array
    #                     )

    #                     covariance_task_dict[wave_block][
    #                         block
    #                     ].simplex_array = xp.vstack(
    #                         (
    #                             covariance_task_dict[wave_block][
    #                                 block
    #                             ].simplex_array,
    #                             new_simplex_array,
    #                         )
    #                     )
    #                     pseudo_covariance_task_dict[wave_block][
    #                         block
    #                     ].simplex_array = xp.vstack(
    #                         (
    #                             pseudo_covariance_task_dict[wave_block][
    #                                 block
    #                             ].simplex_array,
    #                             new_simplex_array,
    #                         )
    #                     )

    #                 elif integration_method == "midpoint":
    #                     old_stack_length = len(
    #                         covariance_task_dict[wave_block][
    #                             block
    #                         ].midpoint_array
    #                     )
    #                     new_stack_length = old_stack_length + len(
    #                         new_midpoint_array
    #                     )

    #                     covariance_task_dict[wave_block][
    #                         block
    #                     ].midpoint_array = xp.vstack(
    #                         (
    #                             covariance_task_dict[wave_block][
    #                                 block
    #                             ].midpoint_array,
    #                             new_midpoint_array,
    #                         )
    #                     )
    #                     covariance_task_dict[wave_block][
    #                         block
    #                     ].volume_array = xp.vstack(
    #                         (
    #                             covariance_task_dict[wave_block][
    #                                 block
    #                             ].volume_array,
    #                             new_volume_array,
    #                         )
    #                     )

    #                     pseudo_covariance_task_dict[wave_block][
    #                         block
    #                     ].midpoint_array = xp.vstack(
    #                         (
    #                             covariance_task_dict[wave_block][
    #                                 block
    #                             ].midpoint_array,
    #                             new_midpoint_array,
    #                         )
    #                     )
    #                     pseudo_covariance_task_dict[wave_block][
    #                         block
    #                     ].volume_array = xp.vstack(
    #                         (
    #                             covariance_task_dict[wave_block][
    #                                 block
    #                             ].volume_array,
    #                             new_volume_array,
    #                         )
    #                     )

    #                 # Add sub block locations.
    #                 # Remember that u and v are negated for pseudo_covariance for
    #                 # reciprocal mode grids
    #                 new_slice = slice(old_stack_length, new_stack_length)

    #                 new_covariance_indices = indices
    #                 new_covariance_sub_block_location = (
    #                     new_slice,
    #                     new_covariance_indices,
    #                 )
    #                 covariance_task_dict[wave_block][
    #                     block
    #                 ].sub_block_locations.append(
    #                     new_covariance_sub_block_location
    #                 )

    #                 new_pseudo_covariance_indices = (i, j, -u, -v)
    #                 new_pseudo_covariance_sub_block_location = (
    #                     new_slice,
    #                     new_pseudo_covariance_indices,
    #                 )
    #                 pseudo_covariance_task_dict[wave_block][
    #                     block
    #                 ].sub_block_locations.append(
    #                     new_pseudo_covariance_sub_block_location
    #                 )

    #                 # Execute tasks if RAM usage is getting too high.
    #                 # Re-initialize relevant integration task
    #                 current_ram_usage = (
    #                     self.integration_task_config.get_current_ram_usage()
    #                 )
    #                 if (
    #                     current_ram_usage
    #                     > self.integration_task_config.ram_limit
    #                 ):
    #                     new_covariance_result = covariance_task_dict[
    #                         wave_block
    #                     ][block].execute_task()
    #                     main_result_list.append_result(new_covariance_result)

    #                     block_location = (wave_block, block)

    #                     covariance_integrand = self._get_covariance_integrand(
    #                         wave_block_one,
    #                         wave_block_two,
    #                         block_one,
    #                         block_two,
    #                     )

    #                     covariance_task = build_integration_task(
    #                         integration_method=integration_method,
    #                         integrand=covariance_integrand,
    #                         statistic_type="covariance",
    #                         block_location=block_location,
    #                         sub_block_locations=[],
    #                         const_factor=const_factor,
    #                         use_cupy=self.integration_task_config.use_gpu,
    #                         cubature_scheme=covariance_cubature_scheme,
    #                         use_dirac_density=False,
    #                     )
    #                     covariance_task_dict[wave_block][
    #                         block
    #                     ] = covariance_task

    #                     new_pseudo_covariance_result = (
    #                         pseudo_covariance_task_dict[wave_block][
    #                             block
    #                         ].execute_task()
    #                     )
    #                     main_result_list.append_result(
    #                         new_pseudo_covariance_result
    #                     )

    #                     block_location = (wave_block, block)

    #                     pseudo_covariance_integrand = (
    #                         self._get_pseudo_covariance_integrand(
    #                             wave_block_one,
    #                             wave_block_two,
    #                             block_one,
    #                             block_two,
    #                         )
    #                     )

    #                     pseudo_covariance_task = build_integration_task(
    #                         integration_method=integration_method,
    #                         integrand=pseudo_covariance_integrand,
    #                         statistic_type="pseudo_covariance",
    #                         block_location=block_location,
    #                         sub_block_locations=[],
    #                         const_factor=const_factor,
    #                         use_cupy=self.integration_task_config.use_gpu,
    #                         cubature_scheme=covariance_cubature_scheme,
    #                         use_dirac_density=False,
    #                     )
    #                     pseudo_covariance_task_dict[wave_block][
    #                         block
    #                     ] = pseudo_covariance_task

    #     # Execute remaining tasks
    #     for wave_block in WAVE_BLOCKS:
    #         for block in self.BLOCKS:
    #             new_covariance_result = covariance_task_dict[wave_block][
    #                 block
    #             ].execute_task()
    #             main_result_list.append_result(new_covariance_result)

    #             new_pseudo_covariance_result = pseudo_covariance_task_dict[
    #                 wave_block
    #             ][block].execute_task()
    #             main_result_list.append_result(new_pseudo_covariance_result)

    #     return main_result_list

    # def _get_covariance_integration_results_eight_dimensions(
    #     self,
    #     covariance_indices: dict[str, dict[str, tuple[int, int, int, int]]],
    #     integration_method: str = "cubature",
    #     covariance_cubature_scheme: Any | None = None,
    # ) -> IntegrationResultList:
    #     """Main method for computing covariance results based on
    #     six dimensional integrals"""
    #     xp = cp if self.integration_task_config.use_gpu else np

    #     # Factor common to all covariance integrals
    #     const_factor = self.medium_parameters.cov_const_factor
    #     main_result_list = IntegrationResultList()
    #     columns_to_keep = [0, 1, 2, 3, 4, 5]

    #     # Prepare task dictionaries
    #     covariance_task_dict = {}
    #     pseudo_covariance_task_dict = {}

    #     for wave_block in WAVE_BLOCKS:
    #         wave_block_one, wave_block_two = wave_block.split(",")

    #         covariance_task_dict[wave_block] = {}
    #         pseudo_covariance_task_dict[wave_block] = {}

    #         for block in self.BLOCKS:
    #             block_one, block_two = block.split(",")
    #             block_location = (wave_block, block)

    #             # Set up covariance and pseudo_covariance separately
    #             covariance_integrand = self._get_covariance_integrand_eight(
    #                 wave_block_one, wave_block_two, block_one, block_two
    #             )
    #             covariance_task = build_integration_task(
    #                 integration_method=integration_method,
    #                 integrand=covariance_integrand,
    #                 statistic_type="covariance",
    #                 block_location=block_location,
    #                 sub_block_locations=[],
    #                 const_factor=const_factor,
    #                 use_cupy=self.integration_task_config.use_gpu,
    #                 cubature_scheme=covariance_cubature_scheme,
    #                 use_dirac_density=False,
    #             )
    #             covariance_task_dict[wave_block][block] = covariance_task

    #             Pseudo_covariance
    #             pseudo_covariance_integrand = (
    #                 self._get_pseudo_covariance_integrand(
    #                     wave_block_one, wave_block_two, block_one, block_two
    #                 )
    #             )
    #             pseudo_covariance_task = build_integration_task(
    #                 integration_method=integration_method,
    #                 integrand=pseudo_covariance_integrand,
    #                 statistic_type="pseudo_covariance",
    #                 block_location=block_location,
    #                 sub_block_locations=[],
    #                 const_factor=const_factor,
    #                 use_cupy=self.integration_task_config.use_gpu,
    #                 cubature_scheme=covariance_cubature_scheme,
    #                 use_dirac_density=False,
    #             )
    #             pseudo_covariance_task_dict[wave_block][
    #                 block
    #             ] = pseudo_covariance_task

    #     # Main loop
    #     master_indices = covariance_indices.get("pp,pp").get("t,t")
    #     mode_vertices_dict = self.mode_grid.propagating_modes_vertices_dict

    #     for indices in self.logger.progress_bar(master_indices):
    #         # Work out the template's integration domain
    #         # This involves the higher dimensional geometry
    #         i, j, u, v = indices
    #         mode_i_vertices = mode_vertices_dict.get(i)
    #         mode_j_vertices = mode_vertices_dict.get(j)
    #         mode_u_vertices = mode_vertices_dict.get(u)
    #         mode_v_vertices = mode_vertices_dict.get(v)

    #         # Get the integration domain
    #         cartesian_product = geometry_utils.iterated_cartesian_product(
    #             [
    #                 mode_i_vertices,
    #                 mode_j_vertices,
    #                 mode_u_vertices,
    #                 mode_v_vertices,
    #             ]
    #         )
    #         reduced_intersection = geometry_utils.get_intersection_vertices(
    #             cartesian_product
    #         )[:, columns_to_keep]
    #         reduced_hull = scipy.spatial.ConvexHull(
    #             reduced_intersection, qhull_options="QJ"
    #         )
    #         centroid = xp.mean(cartesian_product, axis=0)
    #         volume = reduced_hull.volume

    #         if integration_method == "cubature":
    #             centroid_expanded = xp.tile(
    #                 centroid, (reduced_hull.simplices.shape[0], 1, 1)
    #             )
    #             new_simplex_array = xp.concatenate(
    #                 [
    #                     xp.asarray(
    #                         reduced_hull.points[reduced_hull.simplices]
    #                     ),
    #                     centroid_expanded,
    #                 ],
    #                 axis=1,
    #             )
    #         elif integration_method == "midpoint":
    #             new_midpoint_array = xp.array([centroid])
    #             new_volume_array = xp.array([volume])

    #         # Add computed geometric quantities to task dictionaries
    #         for wave_block in WAVE_BLOCKS:
    #             wave_block_one, wave_block_two = wave_block.split(",")
    #             for block in self.BLOCKS:
    #                 block_one, block_two = block.split(",")

    #                 # Check if the mean needs to be calculated for this
    #                 # particular wave_block, block pair
    #                 if indices not in covariance_indices.get(
    #                     wave_block, {}
    #                 ).get(block, set()):
    #                     continue

    #                 # Add domain to integral task
    #                 if integration_method == "cubature":
    #                     old_stack_length = len(
    #                         covariance_task_dict[wave_block][
    #                             block
    #                         ].simplex_array
    #                     )
    #                     new_stack_length = old_stack_length + len(
    #                         new_simplex_array
    #                     )

    #                     covariance_task_dict[wave_block][
    #                         block
    #                     ].simplex_array = xp.vstack(
    #                         (
    #                             covariance_task_dict[wave_block][
    #                                 block
    #                             ].simplex_array,
    #                             new_simplex_array,
    #                         )
    #                     )
    #                     pseudo_covariance_task_dict[wave_block][
    #                         block
    #                     ].simplex_array = xp.vstack(
    #                         (
    #                             pseudo_covariance_task_dict[wave_block][
    #                                 block
    #                             ].simplex_array,
    #                             new_simplex_array,
    #                         )
    #                     )

    #                 elif integration_method == "midpoint":
    #                     old_stack_length = len(
    #                         covariance_task_dict[wave_block][
    #                             block
    #                         ].midpoint_array
    #                     )
    #                     new_stack_length = old_stack_length + len(
    #                         new_midpoint_array
    #                     )

    #                     covariance_task_dict[wave_block][
    #                         block
    #                     ].midpoint_array = xp.vstack(
    #                         (
    #                             covariance_task_dict[wave_block][
    #                                 block
    #                             ].midpoint_array,
    #                             new_midpoint_array,
    #                         )
    #                     )
    #                     covariance_task_dict[wave_block][
    #                         block
    #                     ].volume_array = xp.vstack(
    #                         (
    #                             covariance_task_dict[wave_block][
    #                                 block
    #                             ].volume_array,
    #                             new_volume_array,
    #                         )
    #                     )

    #                     pseudo_covariance_task_dict[wave_block][
    #                         block
    #                     ].midpoint_array = xp.vstack(
    #                         (
    #                             covariance_task_dict[wave_block][
    #                                 block
    #                             ].midpoint_array,
    #                             new_midpoint_array,
    #                         )
    #                     )
    #                     pseudo_covariance_task_dict[wave_block][
    #                         block
    #                     ].volume_array = xp.vstack(
    #                         (
    #                             covariance_task_dict[wave_block][
    #                                 block
    #                             ].volume_array,
    #                             new_volume_array,
    #                         )
    #                     )

    #                 # Add sub block locations.
    #                 # Remember that u and v are negated for pseudo_covariance for
    #                 # reciprocal mode grids
    #                 new_slice = slice(old_stack_length, new_stack_length)

    #                 new_covariance_indices = indices
    #                 new_covariance_sub_block_location = (
    #                     new_slice,
    #                     new_covariance_indices,
    #                 )
    #                 covariance_task_dict[wave_block][
    #                     block
    #                 ].sub_block_locations.append(
    #                     new_covariance_sub_block_location
    #                 )

    #                 new_pseudo_covariance_indices = (i, j, -u, -v)
    #                 new_pseudo_covariance_sub_block_location = (
    #                     new_slice,
    #                     new_pseudo_covariance_indices,
    #                 )
    #                 pseudo_covariance_task_dict[wave_block][
    #                     block
    #                 ].sub_block_locations.append(
    #                     new_pseudo_covariance_sub_block_location
    #                 )

    #                 # Execute tasks if RAM usage is getting too high.
    #                 # Re-initialize relevant integration task
    #                 current_ram_usage = (
    #                     self.integration_task_config.get_current_ram_usage()
    #                 )
    #                 if (
    #                     current_ram_usage
    #                     > self.integration_task_config.ram_limit
    #                 ):
    #                     new_covariance_result = covariance_task_dict[
    #                         wave_block
    #                     ][block].execute_task()
    #                     main_result_list.append_result(new_covariance_result)

    #                     block_location = (wave_block, block)

    #                     covariance_integrand = self._get_covariance_integrand(
    #                         wave_block_one,
    #                         wave_block_two,
    #                         block_one,
    #                         block_two,
    #                     )

    #                     covariance_task = build_integration_task(
    #                         integration_method=integration_method,
    #                         integrand=covariance_integrand,
    #                         statistic_type="covariance",
    #                         block_location=block_location,
    #                         sub_block_locations=[],
    #                         const_factor=const_factor,
    #                         use_cupy=self.integration_task_config.use_gpu,
    #                         cubature_scheme=covariance_cubature_scheme,
    #                         use_dirac_density=False,
    #                     )
    #                     covariance_task_dict[wave_block][
    #                         block
    #                     ] = covariance_task

    #                     new_pseudo_covariance_result = (
    #                         pseudo_covariance_task_dict[wave_block][
    #                             block
    #                         ].execute_task()
    #                     )
    #                     main_result_list.append_result(
    #                         new_pseudo_covariance_result
    #                     )

    #                     block_location = (wave_block, block)

    #                     pseudo_covariance_integrand = (
    #                         self._get_pseudo_covariance_integrand(
    #                             wave_block_one,
    #                             wave_block_two,
    #                             block_one,
    #                             block_two,
    #                         )
    #                     )

    #                     pseudo_covariance_task = build_integration_task(
    #                         integration_method=integration_method,
    #                         integrand=pseudo_covariance_integrand,
    #                         statistic_type="pseudo_covariance",
    #                         block_location=block_location,
    #                         sub_block_locations=[],
    #                         const_factor=const_factor,
    #                         use_cupy=self.integration_task_config.use_gpu,
    #                         cubature_scheme=covariance_cubature_scheme,
    #                         use_dirac_density=False,
    #                     )
    #                     pseudo_covariance_task_dict[wave_block][
    #                         block
    #                     ] = pseudo_covariance_task

    #     # Execute remaining tasks
    #     for wave_block in WAVE_BLOCKS:
    #         for block in self.BLOCKS:
    #             new_covariance_result = covariance_task_dict[wave_block][
    #                 block
    #             ].execute_task()
    #             main_result_list.append_result(new_covariance_result)

    #             new_pseudo_covariance_result = pseudo_covariance_task_dict[
    #                 wave_block
    #             ][block].execute_task()
    #             main_result_list.append_result(new_pseudo_covariance_result)

    #     return main_result_list

    # def _get_covariance_integration_results_six_dimensions_OLD(
    #     self,
    #     class_quadruple_list: ClassQuadrupleList,
    #     covariance_indices: dict[str, dict[str, tuple[int, int, int, int]]],
    #     integration_method: str = "cubature",
    #     covariance_cubature_scheme: Any | None = None,
    # ) -> IntegrationResultList:
    #     """Main method for computing covariance results based on
    #     six dimensional integrals"""
    #     xp = cp if self.integration_task_config.use_gpu else np

    #     # Factor common to all covariance integrals
    #     const_factor = self.medium_parameters.cov_const_factor
    #     main_result_list = IntegrationResultList()
    #     columns_to_keep = [0, 1, 2, 3, 4, 5]
    #     # Prepare task dictionary
    #     task_dict = {}
    #     for wave_block in WAVE_BLOCKS:
    #         wave_block_one, wave_block_two = wave_block.split(",")
    #         task_dict[wave_block] = {}

    #         for block in self.BLOCKS:
    #             block_one, block_two = block.split(",")
    #             block_location = (wave_block, block)

    #             integrand = self._get_covariance_integrand(
    #                 wave_block_one, wave_block_two, block_one, block_two
    #             )

    #             task = build_integration_task(
    #                 integration_method=integration_method,
    #                 integrand=integrand,
    #                 statistic_type="covariance",
    #                 block_location=block_location,
    #                 sub_block_locations=[],
    #                 const_factor=const_factor,
    #                 use_cupy=self.integration_task_config.use_gpu,
    #                 cubature_scheme=covariance_cubature_scheme,
    #                 use_dirac_density=False,
    #             )
    #             task_dict[wave_block][block] = task

    #     # Main loop
    #     for class_number, class_quadruple in enumerate(
    #         self.logger.progress_bar(class_quadruple_list.classes)
    #     ):
    #         # Work out the template's integration domain
    #         # This involves the higher dimensional geometry
    #         template = class_quadruple.template
    #         cartesian_product = template.vertices

    #         # Get the integration domain
    #         reduced_intersection = geometry_utils.get_intersection_vertices(
    #             cartesian_product
    #         )[:, columns_to_keep]

    #         reduced_hull = scipy.spatial.ConvexHull(
    #             reduced_intersection, qhull_options="QJ"
    #         )

    #         # Get the centroid and interweave it into all the simplices
    #         centroid = xp.mean(reduced_intersection, axis=0)
    #         if integration_method == "cubature":
    #             centroid_expanded = xp.tile(
    #                 centroid, (reduced_hull.simplices.shape[0], 1, 1)
    #             )
    #             template_simplex_array = xp.concatenate(
    #                 [
    #                     xp.asarray(
    #                         reduced_hull.points[reduced_hull.simplices]
    #                     ),
    #                     centroid_expanded,
    #                 ],
    #                 axis=1,
    #             )
    #         elif integration_method == "midpoint":
    #             template_midpoint_array = xp.array([centroid])
    #             template_volume_array = xp.array([reduced_hull.volume])

    #         for quadruple in class_quadruple.quadruples:

    #             # ------------
    #             indices = quadruple.singles_indices
    #             start = time.perf_counter()
    #             i, j, u, v = quadruple.singles_indices
    #             mode_i = self.mode_grid.by_index(i).vertices
    #             mode_j = self.mode_grid.by_index(j).vertices
    #             mode_u = self.mode_grid.by_index(u).vertices
    #             mode_v = self.mode_grid.by_index(v).vertices

    #             cartesian_product = geometry_utils.iterated_cartesian_product(
    #                 [mode_i, mode_j, mode_u, mode_v]
    #             )
    #             reduced_intersection = (
    #                 geometry_utils.get_intersection_vertices(
    #                     cartesian_product
    #                 )[:, columns_to_keep]
    #             )

    #             reduced_hull = scipy.spatial.ConvexHull(
    #                 reduced_intersection, qhull_options="QJ"
    #             )
    #             centroid = xp.mean(reduced_intersection, axis=0)
    #             end = time.perf_counter()
    #             # -------------

    #             # indices = quadruple.singles_indices
    #             # new_translation_vector = xp.asarray(
    #             #     quadruple.translation_vector[columns_to_keep]
    #             # )

    #             if integration_method == "cubature":
    #                 new_simplex_array = (
    #                     template_simplex_array + new_translation_vector
    #                 )
    #             elif integration_method == "midpoint":
    #                 # new_midpoint_array = (
    #                 #     template_midpoint_array + new_translation_vector
    #                 # )
    #                 # new_volume_array = np.copy(template_volume_array)
    #                 new_midpoint_array = xp.array([centroid])
    #                 new_volume_array = np.array([reduced_hull.volume])

    #             for wave_block in WAVE_BLOCKS:
    #                 wave_block_one, wave_block_two = wave_block.split(",")
    #                 for block in self.BLOCKS:
    #                     block_one, block_two = block.split(",")
    #                     # Check if the mean needs to be calculated for this
    #                     # particular wave_block, block pair
    #                     if indices not in covariance_indices.get(
    #                         wave_block, {}
    #                     ).get(block, set()):
    #                         continue

    #                     # Add domain to integral task
    #                     if integration_method == "cubature":
    #                         old_stack_length = len(
    #                             task_dict[wave_block][block].simplex_array
    #                         )
    #                         new_stack_length = old_stack_length + len(
    #                             new_simplex_array
    #                         )
    #                         task_dict[wave_block][block].simplex_array = (
    #                             xp.vstack(
    #                                 (
    #                                     task_dict[wave_block][
    #                                         block
    #                                     ].simplex_array,
    #                                     new_simplex_array,
    #                                 )
    #                             )
    #                         )

    #                     elif integration_method == "midpoint":
    #                         old_stack_length = len(
    #                             task_dict[wave_block][block].midpoint_array
    #                         )
    #                         new_stack_length = old_stack_length + len(
    #                             new_midpoint_array
    #                         )
    #                         task_dict[wave_block][block].midpoint_array = (
    #                             xp.vstack(
    #                                 (
    #                                     task_dict[wave_block][
    #                                         block
    #                                     ].midpoint_array,
    #                                     new_midpoint_array,
    #                                 )
    #                             )
    #                         )
    #                         task_dict[wave_block][block].volume_array = (
    #                             xp.vstack(
    #                                 (
    #                                     task_dict[wave_block][
    #                                         block
    #                                     ].volume_array,
    #                                     new_volume_array,
    #                                 )
    #                             )
    #                         )

    #                     new_slice = slice(old_stack_length, new_stack_length)
    #                     new_indices = indices
    #                     new_sub_block_location = (new_slice, new_indices)
    #                     task_dict[wave_block][
    #                         block
    #                     ].sub_block_locations.append(new_sub_block_location)

    #                     # Execute tasks if RAM usage is getting too high.
    #                     # Re-initialize relevant integration task
    #                     current_ram_usage = (
    #                         self.integration_task_config.get_current_ram_usage()
    #                     )
    #                     if (
    #                         current_ram_usage
    #                         > self.integration_task_config.ram_limit
    #                     ):
    #                         new_result = task_dict[wave_block][
    #                             block
    #                         ].execute_task()
    #                         main_result_list.append_result(new_result)

    #                         # Reset task object
    #                         block_location = (wave_block, block)

    #                         # DEBUGGING CODE
    #                         # ---------------------------------------------------
    #                         # ---------------------------------------------------
    #                         # ---------------------------------------------------
    #                         # ---------------------------------------------------
    #                         # ---------------------------------------------------
    #                         # ---------------------------------------------------
    #                         integrand = self._get_covariance_integrand(
    #                             wave_block_one,
    #                             wave_block_two,
    #                             block_one,
    #                             block_two,
    #                         )
    #                         # integrand = self._get_covariance_integrand_symmetric(
    #                         #     wave_block_one,
    #                         #     wave_block_two,
    #                         #     block_one,
    #                         #     block_two,
    #                         # )

    #                         task = build_integration_task(
    #                             integration_method=integration_method,
    #                             integrand=integrand,
    #                             statistic_type="covariance",
    #                             block_location=block_location,
    #                             sub_block_locations=[],
    #                             const_factor=const_factor,
    #                             use_cupy=self.integration_task_config.use_gpu,
    #                             cubature_scheme=covariance_cubature_scheme,
    #                             use_dirac_density=False,
    #                         )
    #                         task_dict[wave_block][block] = task

    #     # Execute remaining tasks
    #     for wave_block in WAVE_BLOCKS:
    #         for block in self.BLOCKS:
    #             new_result = task_dict[wave_block][block].execute_task()
    #             main_result_list.append_result(new_result)
    #     return main_result_list

    # def _get_covariance_integration_results_two_dimensions(
    #     self,
    #     class_quadruple_list: ClassQuadrupleList,
    #     covariance_indices: dict[str, dict[str, tuple[int, int, int, int]]],
    #     covariance_cubature_scheme: Any | None = None,
    # ) -> IntegrationResultList:
    #     """Main method for computing covariance results"""
    #     xp = cp if self.integration_task_config.use_gpu else np

    #     # Factor common to all covariance integrals
    #     const_factor = self.medium_parameters.cov_const_factor
    #     main_result_list = IntegrationResultList()

    #     # Prepare the master index set, which contains all possible indices
    #     # for which a covariance will need to be calculated
    #     master_index_set = set()
    #     for wave_block, d in covariance_indices.items():
    #         for block, i in d.items():
    #             master_index_set.update(i)

    #     # Hyperplane normals
    #     columns_to_keep = [0, 1]
    #     ct = 0

    #     # Main loop
    #     for class_number, class_quadruple in enumerate(
    #         self.logger.progress_bar(class_quadruple_list.classes)
    #     ):
    #         # Work out the template's integration domain
    #         # This involves the higher dimensional geometry
    #         template = class_quadruple.template
    #         cartesian_product = template.vertices

    #         # The 'true" region
    #         reduced_intersection = geometry_utils.get_intersection_vertices(
    #             cartesian_product
    #         )
    #         reduced_hull = scipy.spatial.ConvexHull(
    #             reduced_intersection, qhull_options="QJ"
    #         )
    #         true_volume = reduced_hull.volume

    #         # Dirac density
    #         _, j, _, v = class_quadruple.template.singles_indices
    #         kj = self.mode_grid.by_index(j).center
    #         kv = self.mode_grid.by_index(v).center
    #         intersection = (
    #             geometry_utils.get_intersection_vertices_dirac_density(
    #                 cartesian_product, kj, kv
    #             )
    #         )
    #         geometric_factor = 1
    #         reduced_region = intersection[:, columns_to_keep]
    #         delaunay = scipy.spatial.Delaunay(reduced_region)
    #         new_simplices = xp.asarray(delaunay.points[delaunay.simplices])

    #         for quadruple in class_quadruple.quadruples:
    #             indices = quadruple.singles_indices
    #             new_domain = new_simplices + xp.asarray(
    #                 quadruple.translation_vector[columns_to_keep]
    #             )
    #             _, j, _, v = quadruple.singles_indices
    #             kj = self.mode_grid.by_index(j).center
    #             kv = self.mode_grid.by_index(v).center
    #             dirac_factor = (
    #                 self.medium_parameters.k**4
    #                 * self.mode_grid.by_index(j).weight
    #                 * self.mode_grid.by_index(v).weight
    #             )
    #             new_stack_length = len(new_domain)
    #             new_slice = slice(0, new_stack_length)
    #             new_indices = indices
    #             new_sub_block_location = (new_slice, new_indices)

    #             for wave_block in WAVE_BLOCKS:
    #                 wave_block_one, wave_block_two = wave_block.split(",")
    #                 for block in self.BLOCKS:
    #                     block_one, block_two = block.split(",")

    #                     # Check if the mean needs to be calculated for this
    #                     # particular wave_block, block pair
    #                     if indices not in covariance_indices.get(
    #                         wave_block, {}
    #                     ).get(block, set()):
    #                         continue
    #                     ct += 1

    #                     block_location = (wave_block, block)
    #                     integrand = (
    #                         self._get_covariance_integrand_dirac_density(
    #                             wave_block_one,
    #                             wave_block_two,
    #                             block_one,
    #                             block_two,
    #                             xp.asarray(kj),
    #                             xp.asarray(kv),
    #                         )
    #                     )
    #                     new_task = CubatureIntegrationTask(
    #                         integrand=integrand,
    #                         statistic_type="covariance",
    #                         block_location=(wave_block, block),
    #                         sub_block_locations=[new_sub_block_location],
    #                         const_factor=const_factor,
    #                         use_cupy=self.integration_task_config.use_gpu,
    #                         simplex_array=new_domain,
    #                         cubature_scheme=covariance_cubature_scheme,
    #                         use_dirac_density=True,
    #                     )

    #                     new_result = new_task.execute_task()
    #                     new_result.integral = (
    #                         new_result.integral * dirac_factor
    #                     )
    #                     main_result_list.append_result(new_result)

    #     return main_result_list

    # def _get_pseudo_covariance_integration_tasks(
    #     self,
    #     quadruples,
    #     independent_elements: dict[str, dict[str, tuple[int, int, int, int]]],
    # ) -> IntegrationTaskList:
    #     # Multiprocessing parameters
    #     num_quadruples = len(quadruples)
    #     num_processes = min(num_quadruples, os.cpu_count())

    #     # Prepare integrands since these can't be pickled
    #     blocks = [
    #         "t,t",
    #         "t,r",
    #         "t,t2",
    #         "t,r2",
    #         "r,r",
    #         "r,t2",
    #         "r,r2",
    #         "t2,t2",
    #         "t2,r2",
    #         "r2,r2",
    #     ]

    #     integrand = {}

    #     for key in blocks:
    #         block_one, block_two = key.split(",")
    #         integrand[key] = self._get_pseudo_covariance_integrand(
    #             "pp", "pp", block_one, block_two
    #         )

    #     parallelised_function = functools.partial(
    #         self._get_pseudo_covariance_integration_tasks_partial,
    #         independent_elements=independent_elements,
    #         const_factor=self.medium_parameters.cov_const_factor,
    #         integrals_per_task=self.integration_task_config.integrals_per_task,
    #         integrand=integrand,
    #         progress_bar=self.logger.progress_bar,
    #     )

    #     partial_quadruples = array_utils.split_list(quadruples, num_processes)

    #     with ProcessPool(processes=num_processes) as pool:
    #         out = pool.map(parallelised_function, partial_quadruples)

    #     main_result_list = IntegrationResultList()

    #     for result_list in out:
    #         main_result_list.merge_result_list(result_list)

    #     return main_result_list

    # @staticmethod
    # def _get_pseudo_covariance_integration_tasks_partial(
    #     quadruples,
    #     independent_elements: dict[str, dict[str, tuple[int, int, int, int]]],
    #     const_factor,
    #     integrals_per_task,
    #     integrand,
    #     progress_bar,
    # ) -> IntegrationTaskList:
    #     """Main method for preparing covariance integral tasks"""

    #     # Factor common to all covariance integrals
    #     main_result_list = IntegrationResultList()

    #     # wave block will look like "pp,pp", "pp,ep" etc.
    #     wave_block = "pp,pp"
    #     wave_block_one, wave_block_two = wave_block.split(",")

    #     # # The integrand depends only on the matrix blocks.

    #     # Variables that will be used for constructing tasks
    #     # These reset each time we go to a new block
    #     blocks = [
    #         "t,t",
    #         "t,r",
    #         "t,t2",
    #         "t,r2",
    #         "r,r",
    #         "r,t2",
    #         "r,r2",
    #         "t2,t2",
    #         "t2,r2",
    #         "r2,r2",
    #     ]

    #     # Initialise empty dictionaries for storing all the data that will be
    #     # used in the ensuing loop
    #     sub_block_locations = {}
    #     simplex_stack = {}
    #     stack_length = {}
    #     integration_results = {}

    #     for key in blocks:
    #         sub_block_locations[key] = []
    #         simplex_stack[key] = np.zeros((0, 7, 6), dtype=np.float64)
    #         stack_length[key] = 0
    #         integration_results[key] = IntegrationResultList()
    #         block_one, block_two = key.split(",")

    #     # Main loop
    #     for quadruple in progress_bar(quadruples):
    #         # Asses which blocks this particular quadruple will have statistics
    #         # for
    #         i, j, u, v = quadruple.singles
    #         first = (i, j)
    #         second = (u, v)

    #         valid_blocks = ["t,t"]
    #         if second in independent_elements["r"]:
    #             valid_blocks.append("t,r")
    #         if second in independent_elements["t2"]:
    #             valid_blocks.append("t,t2")
    #         if second in independent_elements["r2"]:
    #             valid_blocks.append("t,r2")

    #         if first in independent_elements["r"]:
    #             if second in independent_elements["r"]:
    #                 valid_blocks.append("r,r")
    #             if second in independent_elements["t2"]:
    #                 valid_blocks.append("r,t2")
    #             if second in independent_elements["r2"]:
    #                 valid_blocks.append("r,r2")

    #         if first in independent_elements["t2"]:
    #             if second in independent_elements["t2"]:
    #                 valid_blocks.append("t2,t2")
    #             if second in independent_elements["r2"]:
    #                 valid_blocks.append("t2,t2")

    #         if first in independent_elements["r2"]:
    #             if second in independent_elements["r2"]:
    #                 valid_blocks.append("r2,r2")

    #         # Check how long the stack will become if the new triangles
    #         # are added. If this length exceeds the limit, we begin
    #         # working on a new task
    #         new_simplices = quadruple.domain

    #         for block in valid_blocks:
    #             new_stack_length = stack_length[block] + len(new_simplices)

    #             if (
    #                 integrals_per_task is not None
    #                 and stack_length[block] > 0
    #                 and new_stack_length > integrals_per_task
    #             ):
    #                 new_task = IntegrationTask(
    #                     integrand[block],
    #                     simplex_stack[block],
    #                     statistic_type="pseudo_covariance",
    #                     block_location=(wave_block, block),
    #                     sub_block_locations=sub_block_locations[block],
    #                     const_factor=const_factor,
    #                 )
    #                 new_result = new_task.execute_task()
    #                 integration_results[block].append_result(new_result)

    #                 # Reset the triangle stack and stack length
    #                 simplex_stack[block] = np.zeros(
    #                     (0, 7, 6), dtype=np.float64
    #                 )
    #                 stack_length[block] = 0
    #                 sub_block_locations[block] = []

    #             # Add location to sub_block_locations
    #             new_slice = slice(stack_length[block], new_stack_length)
    #             new_indices = (i, j, -u, -v)
    #             new_sub_block_location = (new_slice, new_indices)
    #             sub_block_locations[block].append(new_sub_block_location)

    #             # Add new triangles to stack
    #             simplex_stack[block] = np.vstack(
    #                 (simplex_stack[block], new_simplices)
    #             )
    #             stack_length[block] += len(new_simplices)

    #     # Once this point has been reached, we have exahusted all
    #     # triangles for a certing block of the scattering matrix
    #     # We now make the final task for the group
    #     for block in blocks:
    #         new_task = IntegrationTask(
    #             integrand[block],
    #             simplex_stack[block],
    #             statistic_type="pseudo_covariance",
    #             block_location=(wave_block, block),
    #             sub_block_locations=sub_block_locations[block],
    #             const_factor=const_factor,
    #         )
    #         new_result = new_task.execute_task()
    #         integration_results[block].append_result(new_result)
    #         main_result_list.merge_result_list(integration_results[block])

    #     return main_result_list


    def _get_covariance_results_lattice_generator(
        self,
        covariance_indices: dict[str, dict[str, tuple[int, int, int, int]]],
    ) -> IntegrationResultList:
        """Get all the covariance results for lattice based mode_grids
        where memory effect is desired. This uses several appoximations to
        make the computation time faesible."""

        xp = cp if self.integration_task_config.use_gpu else np
        k = self.medium_parameters.k
        L = self.medium_parameters.L
        const_factor = self.medium_parameters.cov_const_factor

        # Pre-compute all A matrices
        num_modes = self.mode_grid.num_propagating
        mode_vertices_dict = self.mode_grid.propagating_modes_vertices_dict
        mean_mode_vertices_dict = (
            self.mode_grid.propagating_modes_mean_vertices_dict
        )
        mean_mode_vertices_array = np.stack(mean_mode_vertices_dict.values())
        ki_array = np.repeat(mean_mode_vertices_array, num_modes, axis=0)
        ki_x_array, ki_y_array = ki_array[:, 0], ki_array[:, 1]
        ki_z_array = np.sqrt(1.0 - ki_x_array**2 - ki_y_array**2)
        kj_array = np.tile(mean_mode_vertices_array, (num_modes, 1))
        kj_x_array, kj_y_array = kj_array[:, 0], kj_array[:, 1]
        kj_z_array = np.sqrt(1.0 - kj_x_array**2 - kj_y_array**2)

        # A matrix values. p and m refere to positive or negative signs on
        # k_z
        get_A = self.medium_statistics.get_mean_a_matrix()
        A_matrix_values_pp = get_A(
            ki_x_array,
            ki_y_array,
            ki_z_array,
            kj_x_array,
            kj_y_array,
            kj_z_array,
        )
        A_matrix_values_pm = get_A(
            ki_x_array,
            ki_y_array,
            ki_z_array,
            kj_x_array,
            kj_y_array,
            -kj_z_array,
        )
        A_matrix_values_mp = get_A(
            ki_x_array,
            ki_y_array,
            -ki_z_array,
            kj_x_array,
            kj_y_array,
            kj_z_array,
        )
        A_matrix_values_mm = get_A(
            ki_x_array,
            ki_y_array,
            -ki_z_array,
            kj_x_array,
            kj_y_array,
            -kj_z_array,
        )

        # Find the volume associated with no shift
        columns_to_keep = [0, 1, 2, 3, 4, 5]
        repeating_mode_vertices = mode_vertices_dict.get(0)
        cartesian_product = geometry_utils.iterated_cartesian_product(
            [
                repeating_mode_vertices,
                repeating_mode_vertices,
                repeating_mode_vertices,
                repeating_mode_vertices,
            ]
        )
        reduced_intersection = geometry_utils.get_intersection_vertices(
            cartesian_product
        )[:, columns_to_keep]
        reduced_hull = scipy.spatial.ConvexHull(
            reduced_intersection, qhull_options="QJ"
        )
        repeating_volume = reduced_hull.volume

        # Begin main quadruple index loop
        master_indices = covariance_indices.get("pp,pp").get("t,t")
        reciprocity_correction = int((num_modes - 1) // 2)
        for indices in self.logger.progress_bar(master_indices):
            i, j, u, v = indices

            # Determine domain volume. If it's an auto-correlation, it might
            # be an edge mode, which will have smaller area.
            if i == u and j == v:
                mode_i_vertices = mode_vertices_dict.get(i)
                mode_j_vertices = mode_vertices_dict.get(j)
                mode_u_vertices = mode_vertices_dict.get(u)
                mode_v_vertices = mode_vertices_dict.get(v)
                cartesian_product = geometry_utils.iterated_cartesian_product(
                    [
                        mode_i_vertices,
                        mode_j_vertices,
                        mode_u_vertices,
                        mode_v_vertices,
                    ]
                )
                reduced_intersection = (
                    geometry_utils.get_intersection_vertices(
                        cartesian_product
                    )[:, columns_to_keep]
                )
                reduced_hull = scipy.spatial.ConvexHull(
                    reduced_intersection, qhull_options="QJ"
                )
                volume = reduced_hull.volume
            else:
                volume = repeating_volume

            # Calculate integral
            ij_val = (
                num_modes * (i + reciprocity_correction)
                + j
                + reciprocity_correction
            )
            uv_val = (
                num_modes * (u + reciprocity_correction)
                + v
                + reciprocity_correction
            )
            ki_z = ki_z_array[ij_val]
            kj_z = kj_z_array[ij_val]
            ku_z = ki_z_array[uv_val]
            kv_z = kj_z_array[uv_val]
            sec_factor = 1.0 / xp.sqrt(xp.abs(ki_z * kj_z * ku_z * kv_z))

            # -----------------------------------------------------------------
            # t,t

            A_ij = A_matrix_values_pp[ij_val]
            A_uv = A_matrix_values_pp[uv_val]
            sinc_factor = special_functions.sinc(
                k * L * (ki_z - kj_z - ku_z + kv_z)
            )
            covariance = (
                volume
                * const_factor
                * sec_factor
                * sinc_factor
                * np.outer(A_ij, A_uv.conj()).ravel()
            )[np.newaxis, :]
            new_result = IntegrationResult(
                "covariance", ("pp,pp", "t,t"), [(i, j, u, v)], covariance
            )
            yield new_result

            # -----------------------------------------------------------------
            # r,r

            A_ij = A_matrix_values_pm[ij_val]
            A_uv = A_matrix_values_pm[uv_val]
            sinc_factor = special_functions.sinc(
                k * L * (ki_z + kj_z - ku_z - kv_z)
            )
            integral = (
                volume
                * const_factor
                * sec_factor
                * sinc_factor
                * np.outer(A_ij, A_uv.conj().T).ravel()
            )[np.newaxis, :]
            new_result = IntegrationResult(
                "covariance", ("pp,pp", "r,r"), [(i, j, u, v)], integral
            )
            yield new_result

            # -----------------------------------------------------------------
            # t2, t2

            A_ij = A_matrix_values_mm[ij_val]
            A_uv = A_matrix_values_mm[uv_val]
            sinc_factor = special_functions.sinc(
                k * L * (-ki_z + kj_z + ku_z - kv_z)
            )
            integral = (
                volume
                * const_factor
                * sec_factor
                * sinc_factor
                * np.outer(A_ij, A_uv.conj().T).ravel()
            )[np.newaxis, :]
            new_result = IntegrationResult(
                "covariance", ("pp,pp", "t2,t2"), [(i, j, u, v)], integral
            )
            yield new_result

            # -----------------------------------------------------------------
            # r2,r2

            A_ij = A_matrix_values_mp[ij_val]
            A_uv = A_matrix_values_mp[uv_val]
            sinc_factor = special_functions.sinc(
                k * L * (-ki_z - kj_z + ku_z + kv_z)
            )
            integral = (
                volume
                * const_factor
                * sec_factor
                * sinc_factor
                * np.outer(A_ij, A_uv.conj().T).ravel()
            )[np.newaxis, :]
            new_result = IntegrationResult(
                "covariance", ("pp,pp", "r2,r2"), [(i, j, u, v)], integral
            )
            yield new_result

    def _get_covariance_results_lattice(
        self,
        covariance_indices: dict[str, dict[str, tuple[int, int, int, int]]],
    ) -> IntegrationResultList:
        """Get all the covariance results for lattice based mode_grids
        where memory effect is desired. This uses several appoximations to
        make the computation time faesible."""

        xp = cp if self.integration_task_config.use_gpu else np
        main_result_list = IntegrationResultList()
        k = self.medium_parameters.k
        L = self.medium_parameters.L
        const_factor = self.medium_parameters.cov_const_factor

        # Pre-compute all A matrices
        num_modes = self.mode_grid.num_propagating
        mode_vertices_dict = self.mode_grid.propagating_modes_vertices_dict
        mean_mode_vertices_dict = (
            self.mode_grid.propagating_modes_mean_vertices_dict
        )
        mean_mode_vertices_array = np.stack(mean_mode_vertices_dict.values())
        ki_array = np.repeat(mean_mode_vertices_array, num_modes, axis=0)
        ki_x_array, ki_y_array = ki_array[:, 0], ki_array[:, 1]
        ki_z_array = np.sqrt(1.0 - ki_x_array**2 - ki_y_array**2)
        kj_array = np.tile(mean_mode_vertices_array, (num_modes, 1))
        kj_x_array, kj_y_array = kj_array[:, 0], kj_array[:, 1]
        kj_z_array = np.sqrt(1.0 - kj_x_array**2 - kj_y_array**2)

        # A matrix values. p and m refere to positive or negative signs on
        # k_z
        get_A = self.medium_statistics.get_mean_a_matrix()
        A_matrix_values_pp = get_A(
            ki_x_array,
            ki_y_array,
            ki_z_array,
            kj_x_array,
            kj_y_array,
            kj_z_array,
        )
        A_matrix_values_pm = get_A(
            ki_x_array,
            ki_y_array,
            ki_z_array,
            kj_x_array,
            kj_y_array,
            -kj_z_array,
        )
        A_matrix_values_mp = get_A(
            ki_x_array,
            ki_y_array,
            -ki_z_array,
            kj_x_array,
            kj_y_array,
            kj_z_array,
        )
        A_matrix_values_mm = get_A(
            ki_x_array,
            ki_y_array,
            -ki_z_array,
            kj_x_array,
            kj_y_array,
            -kj_z_array,
        )

        # Find the volume associated with no shift
        columns_to_keep = [0, 1, 2, 3, 4, 5]
        repeating_mode_vertices = mode_vertices_dict.get(0)
        cartesian_product = geometry_utils.iterated_cartesian_product(
            [
                repeating_mode_vertices,
                repeating_mode_vertices,
                repeating_mode_vertices,
                repeating_mode_vertices,
            ]
        )
        reduced_intersection = geometry_utils.get_intersection_vertices(
            cartesian_product
        )[:, columns_to_keep]
        reduced_hull = scipy.spatial.ConvexHull(
            reduced_intersection, qhull_options="QJ"
        )
        repeating_volume = reduced_hull.volume

        # Begin main quadruple index loop
        master_indices = covariance_indices.get("pp,pp").get("t,t")
        reciprocity_correction = int((num_modes - 1) // 2)
        for indices in self.logger.progress_bar(master_indices):
            i, j, u, v = indices

            # Determine domain volume. If it's an auto-correlation, it might
            # be an edge mode, which will have smaller area.
            if i == u and j == v:
                mode_i_vertices = mode_vertices_dict.get(i)
                mode_j_vertices = mode_vertices_dict.get(j)
                mode_u_vertices = mode_vertices_dict.get(u)
                mode_v_vertices = mode_vertices_dict.get(v)
                cartesian_product = geometry_utils.iterated_cartesian_product(
                    [
                        mode_i_vertices,
                        mode_j_vertices,
                        mode_u_vertices,
                        mode_v_vertices,
                    ]
                )
                reduced_intersection = (
                    geometry_utils.get_intersection_vertices(
                        cartesian_product
                    )[:, columns_to_keep]
                )
                reduced_hull = scipy.spatial.ConvexHull(
                    reduced_intersection, qhull_options="QJ"
                )
                volume = reduced_hull.volume
            else:
                volume = repeating_volume

            # Calculate integral
            ij_val = (
                num_modes * (i + reciprocity_correction)
                + j
                + reciprocity_correction
            )
            uv_val = (
                num_modes * (u + reciprocity_correction)
                + v
                + reciprocity_correction
            )
            ki_z = ki_z_array[ij_val]
            kj_z = kj_z_array[ij_val]
            ku_z = ki_z_array[uv_val]
            kv_z = kj_z_array[uv_val]
            sec_factor = 1.0 / xp.sqrt(xp.abs(ki_z * kj_z * ku_z * kv_z))

            # -----------------------------------------------------------------
            # t,t

            A_ij = A_matrix_values_pp[ij_val]
            A_uv = A_matrix_values_pp[uv_val]
            sinc_factor = special_functions.sinc(
                k * L * (ki_z - kj_z - ku_z + kv_z)
            )
            covariance = (
                volume
                * const_factor
                * sec_factor
                * sinc_factor
                * np.outer(A_ij, A_uv.conj()).ravel()
            )[np.newaxis, :]
            new_result = IntegrationResult(
                "covariance", ("pp,pp", "t,t"), [(i, j, u, v)], covariance
            )
            main_result_list.append_result(new_result)

            pseudo_covariance = (
                volume
                * const_factor
                * sec_factor
                * sinc_factor
                * np.outer(A_ij, A_uv).ravel()
            )[np.newaxis, :]
            new_result = IntegrationResult(
                "pseudo_covariance",
                ("pp,pp", "t,t"),
                [(i, j, u, v)],
                pseudo_covariance,
            )
            main_result_list.append_result(new_result)

            # -----------------------------------------------------------------
            # r,r

            A_ij = A_matrix_values_pm[ij_val]
            A_uv = A_matrix_values_pm[uv_val]
            sinc_factor = special_functions.sinc(
                k * L * (ki_z + kj_z - ku_z - kv_z)
            )
            integral = (
                volume
                * const_factor
                * sec_factor
                * sinc_factor
                * np.outer(A_ij, A_uv.conj().T).ravel()
            )[np.newaxis, :]
            new_result = IntegrationResult(
                "covariance", ("pp,pp", "r,r"), [(i, j, u, v)], integral
            )
            main_result_list.append_result(new_result)

            # -----------------------------------------------------------------
            # t2, t2

            A_ij = A_matrix_values_mm[ij_val]
            A_uv = A_matrix_values_mm[uv_val]
            sinc_factor = special_functions.sinc(
                k * L * (-ki_z + kj_z + ku_z - kv_z)
            )
            integral = (
                volume
                * const_factor
                * sec_factor
                * sinc_factor
                * np.outer(A_ij, A_uv.conj().T).ravel()
            )[np.newaxis, :]
            new_result = IntegrationResult(
                "covariance", ("pp,pp", "t2,t2"), [(i, j, u, v)], integral
            )
            main_result_list.append_result(new_result)

            # -----------------------------------------------------------------
            # r2,r2

            A_ij = A_matrix_values_mp[ij_val]
            A_uv = A_matrix_values_mp[uv_val]
            sinc_factor = special_functions.sinc(
                k * L * (-ki_z - kj_z + ku_z + kv_z)
            )
            integral = (
                volume
                * const_factor
                * sec_factor
                * sinc_factor
                * np.outer(A_ij, A_uv.conj().T).ravel()
            )[np.newaxis, :]
            new_result = IntegrationResult(
                "covariance", ("pp,pp", "r2,r2"), [(i, j, u, v)], integral
            )
            main_result_list.append_result(new_result)

        return main_result_list

    def _get_mean_S_vector(
        self, mean_result_list: integration_task.IntegrationResultList
    ) -> Numeric:
        """Construct the mean scattering matrix from the mean results list

        Experiments suggest that this is slower than the other method!
        """

        size_of_S = 4 * self.mode_grid.num_propagating
        mean_S = np.zeros((size_of_S, size_of_S), dtype=np.complex128)
        half_index = int(size_of_S / 2)

        for result in mean_result_list.results:
            # Skip if no new results
            if len(result.sub_block_locations) == 0:
                continue

            wave_block, block = result.block_location
            sub_block_locations = np.array(result.sub_block_locations)

            # Account for reciprocity
            if self.mode_grid.is_reciprocal:
                sub_block_locations += int(
                    (self.mode_grid.num_propagating - 1) / 2
                )

            sub_block_locations *= 2

            # Make copies
            add_y = np.copy(sub_block_locations)
            add_y[:, 1] += 1

            add_x = np.copy(sub_block_locations)
            add_x[:, 0] += 1

            add_xy = np.copy(sub_block_locations)
            add_xy += 1

            sub_block_locations = np.hstack(
                [sub_block_locations, add_y, add_x, add_xy]
            ).reshape((4 * len(add_y), 2))

            # Account for block
            match block:
                case "r":
                    pass
                case "t":
                    sub_block_locations[:, 0] += half_index
                case "t2":
                    sub_block_locations[:, 1] += half_index
                case "r2":
                    sub_block_locations += half_index

            mean_S[sub_block_locations[:, 0], sub_block_locations[:, 1]] = (
                np.ravel(result.integral)
            )

        mean_S_sym = self.mode_grid.rec_mat @ mean_S.T @ self.mode_grid.rec_mat
        mean_S = mean_S + mean_S_sym

        # Multiply by weights
        mean_weight_matrix = self._get_mean_weight_matrix()
        mean_S = mean_weight_matrix @ mean_S @ mean_weight_matrix

        return mean_S