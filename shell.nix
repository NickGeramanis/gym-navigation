{ pkgs ? import <nixpkgs> {} }:
let
  gymNavigation = pkgs.python313Packages.buildPythonPackage {
    pname = "gym_navigation";
    version = "1.0.0";
    src = ./.;
    format = "setuptools";
    propagatedBuildInputs = with pkgs.python313Packages; [
      gymnasium
      numpy
      pygame
    ];
  };
in
pkgs.mkShell {
  buildInputs = [
    (pkgs.python313.withPackages (ps: with ps; [
      gymNavigation
      pytest
      coverage
      pylint
      mypy
    ]))
  ];
}
