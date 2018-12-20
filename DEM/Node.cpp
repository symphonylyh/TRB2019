/**
 * @file Node.cpp
 * Implementation of Node class.
 *
 * @author Haohang Huang
 * @date Feburary 4, 2018
 */

#include "Node.h"

using namespace Eigen;

Node::Node()
{
}

Node::Node(const int & index, const double & x, const double & y)
  : index_(index), globalCoord_(x,y), disp_(0,0), strain_(VectorXd::Zero(4)),
    stress_(VectorXd::Zero(4)), averageCount_(0)
{ // All using initializer list
}

Node::Node(Node const & other)
{
    copy_(other);
}

Node const & Node::operator=(Node const & other)
{
    if (this != &other) {
      clear_();
      copy_(other);
    }
    return *this;
}

Node::~Node()
{
    clear_();
}

void Node::clear_()
{ // No dynamically allocated memory within Node class, so do nothing
}

void Node::copy_(Node const & other)
{
    index_ = other.index_;
    globalCoord_ = other.globalCoord_;
    disp_ = other.disp_;
    force_ = other.force_;
    strain_ = other.strain_;
    stress_ = other.stress_;
    averageCount_ = other.averageCount_;
}

void Node::setGlobalCoord(const double & x, const double & y)
{
    globalCoord_ << x, y;
}

void Node::setDisp(const double & u, const double & v)
{
    disp_ << u, v;
}

void Node::setForce(const double & Fx, const double & Fy)
{
    force_ << Fx, Fy;
}

void Node::setStrainAndStress(const VectorXd & strain, const VectorXd & stress)
{
    strain_ += strain;
    stress_ += stress;
    averageCount_++;
}

const VectorXd & Node::averageStrain() {
    strain_ /= averageCount_;
    return strain_;
}

const VectorXd & Node::averageStress() {
    stress_ /= averageCount_;
    return stress_;
}

const int & Node::getIndex() const
{
    return index_;
}

const Vector2d & Node::getGlobalCoord() const
{
    return globalCoord_;
}

const Vector2d & Node::getDisp() const
{
    return disp_;
}

const Vector2d & Node::getForce() const
{
    return force_;
}
