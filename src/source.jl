# Create elastic wave sources
import MultipleScattering: point_source, AbstractSource
"""
    plane_z_source(medium::Elastic{3,T}, pos::AbstractArray{T}, amplitude::Union{T,Complex{T}}

Creates an incident plane wave for the displacement in the form ``A e^{i k z} (0,1,0)``. The coefficients of the Debye potentials which correspond to this plane wave are given in "Resonance theory of elastic waves ultrasonically scattered from an elastic sphere - 1987".
"""
struct BearingSource{Dim,T}
    source_position::Vector{T}
    amplitudes::Vector{T}
    potentials::Vector{H} where H <: HelmholtzPotential{Dim,T}
    
    function BearingSource(source_position::Vector{T}, amplitudes::Vector{T}, potentials::Vector{H}) where {Dim,T,H<:HelmholtzPotential{Dim,T}}
        new{Dim,T}(source_position, amplitudes, potentials)
    end

end

function point_source(bearing::RollerBearing{T}, source_position::Vector{T}, amplitudes::Vector{T}, modes::AbstractVector{Int}, ω::T) where T
    
    medium = bearing.medium
    kp = ω/medium.cp
    ks = ω/medium.cs
    medium_p = Acoustic(medium.ρ, medium.cp, 2)
    medium_s = Acoustic(medium.ρ, medium.cs, 2)
    order = modes[length(modes)]
    
    coes_inner = (im/4)*hcat([amplitudes[1]*outgoing_translation_matrix(medium_p, 0, order, ω, -source_position), zeros(ComplexF64, length(modes)), amplitudes[2]*outgoing_translation_matrix(medium_s, 0, order, ω, -source_position), zeros(ComplexF64, length(modes))]...) |>collect |>transpose

    coes_outer = (im/4)*hcat([zeros(ComplexF64, length(modes)), amplitudes[1]*regular_translation_matrix(medium_p, 0, order, ω, -source_position), zeros(ComplexF64, length(modes)), amplitudes[2]*regular_translation_matrix(medium_s, 0, order, ω, -source_position)]...) |>collect |>transpose
    
    M0s_inner = [vcat(boundarycondition_mode(ω,TractionBoundary(inner=true), bearing, n), zeros(ComplexF64, 2,4)) for n in modes]
    
    M0s_outer = [vcat(zeros(ComplexF64, 2,4), boundarycondition_mode(ω,TractionBoundary(outer=true), bearing, n)) for n in modes]

    Ms = [boundarycondition_system(ω, bearing, TractionBoundary(inner=true), TractionBoundary(outer=true), n) for n in modes]

    τ_r1 = [-M0s_inner[i]*coes_inner[:,i] for i in 1:length(modes)]
    τ_r2 = [-M0s_outer[i]*coes_outer[:,i] for i in 1:length(modes)]

    reflected_coes = [Ms[i] \ (τ_r1[i]+τ_r2[i]) for i in 1:length(modes)]
    reflected_coes = hcat(reflected_coes...) |>collect

    ϕ = HelmholtzPotential{2}(medium.cp, kp, reflected_coes[1:2,:], modes)
    ψ = HelmholtzPotential{2}(medium.cs, ks, reflected_coes[3:4,:], modes)

    potentials = [ϕ,ψ]
    return BearingSource(source_position, amplitudes, potentials)
end

function pressure_point_source(medium::Elastic{2,T}, source_position::AbstractVector, amplitude::Union{T,Complex{T},Function} = one(T))::RegularSource{Elastic{2,T}} where T <: AbstractFloat

    # Convert to SVector for efficiency and consistency
    source_position = SVector{2,T}(source_position)

    if typeof(amplitude) <: Number
        amp(ω) = amplitude
    else
        amp = amplitude
    end
    source_field(x,ω) = (amp(ω)*im)/4 * hankelh1(0,ω/medium.cp * norm(x-source_position))

    function source_coef(order,centre,ω)
        k = ω/medium.cp
        r, θ = cartesian_to_radial_coordinates(centre - source_position)

        # using Graf's addition theorem
        return (amp(ω)*im)/4 * [hankelh1(-n,k*r) * exp(-im*n*θ) for n = -order:order]
    end

    return RegularSource{Elastic{2,T},WithoutSymmetry{2}}(medium, source_field, source_coef)
end

function shear_point_source(medium::Elastic{2,T}, source_position::AbstractVector, amplitude::Union{T,Complex{T},Function} = one(T))::RegularSource{Elastic{2,T}} where T <: AbstractFloat

    # Convert to SVector for efficiency and consistency
    source_position = SVector{2,T}(source_position)

    if typeof(amplitude) <: Number
        amp(ω) = amplitude
    else
        amp = amplitude
    end
    source_field(x,ω) = (amp(ω)*im)/4 * hankelh1(0,ω/medium.cs * norm(x-source_position))

    function source_coef(order,centre,ω)
        k = ω/medium.cs
        r, θ = cartesian_to_radial_coordinates(centre - source_position)

        # using Graf's addition theorem
        return (amp(ω)*im)/4 * [hankelh1(-n,k*r) * exp(-im*n*θ) for n = -order:order]
    end

    return RegularSource{Elastic{2,T},WithoutSymmetry{2}}(medium, source_field, source_coef)
end

function plane_z_shear_source(medium::Elastic{3,T}, pos::AbstractArray{T} = zeros(T,3),
            amplitude::Union{T,Complex{T}} = one(T)
        ) where {T}

    S = PlanarSymmetry{3}

    # code assumes wave propagates in z-direction and polarised in y-direction     
    direction = [zero(T),zero(T),one(T)]
    direction = direction / norm(direction)

    polarisation = [one(T), zero(T),zero(T)]
    polarisation = polarisation / norm(polarisation)

    # Convert to SVector for efficiency and consistency
    # position = SVector(position...)

    function source_field(x,ω)
        # x_width = norm((x - position) - dot(x - position, direction)*direction)
        
        return amplitude .* exp(im * ω / medium.cs * dot(x - pos, direction)) .* polarisation
    end

    function spherical_expansion(order,centre,ω)
        
        ks = ω / medium.cs
        ks2 = ks^2

        pcoefs = [Complex{T}(0.0) for l = 0:order for m = -l:l] 
        # p_potential = HelmholtzPotential{3}(medium.cp, ω / medium.cp, [pcoefs; 0 .* pcoefs])
        
        Φcoefs = T(sqrt(pi)) * sum(source_field(centre,ω) .* polarisation) .*
        [
            (abs(m) == 1) ?  Complex{T}(m * (1.0im)^l * sqrt(T(2l + 1) / T((1+l)*l))) : Complex{T}(0)
        for l = 0:order for m = -l:l] 
        # Φ_potential = HelmholtzPotential{3}(medium.cs, ω / medium.cs, [Φcoefs; 0 .* Φcoefs])
        
        χcoefs = T(sqrt(pi)) * sum(source_field(centre,ω) .* polarisation) .*
        [
            (abs(m) == 1) ?  Complex{T}((1.0im)^l * sqrt(T(2l + 1) / T((1+l)*l))) : Complex{T}(0)
        for l = 0:order for m = -l:l] 
        # χ_potential = HelmholtzPotential{3}(medium.cs, ω / medium.cs, [χcoefs; 0 .* χcoefs])

        # return ElasticWave(ω, medium, [pcoefs,Φcoefs,χcoefs])
        return [pcoefs Φcoefs χcoefs] ./ (-im * ks) |> transpose
    end

    return RegularSource{Elastic{3,T},S}(medium, source_field, spherical_expansion)
end

# vs = regular_basis_function(source.medium, ω)
# regular_coefficients = regular_spherical_coefficients(source)

# for x close to centre
# source_field(x,ω) ~= sum(regular_coefficients(basis_order,centre,ω) .* vs(basis_order, x - centre))
