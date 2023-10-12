/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2014, Rice University
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Rice University nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/* Author: Ryan Luna */

#ifndef OMPL_GEOMETRIC_CONSTRAINTS_END_EFFECTOR_CONSTRAINT_
#define OMPL_GEOMETRIC_CONSTRAINTS_END_EFFECTOR_CONSTRAINT_

#include <vector>
#include <limits>
#include <functional>
#include <Eigen/Dense>

#include "ompl/base/Constraint.h"
#include "ompl/util/RandomNumbers.h"

namespace ompl
{
    namespace geometric
    {
        /// @cond IGNORE
        OMPL_CLASS_FORWARD(EndEffectorConstraint);
        /// @endcond

        /// \brief A function definition for forward kinematics.  For the given state,
        /// compute the global reference frame of each link in the kinematic chain
        /// Assumed that the frame return is the end effector
        typedef std::function<void(const base::State*, Eigen::Affine3d&)> EEForwardKinematicsFn;
        //typedef boost::function<void(const base::State*, std::vector<Eigen::Affine3d>&)> EEForwardKinematicsFn;

        /// \brief A function definition for inverse kinematics.  For the given pose,
        /// compute the joint positions for the kinematic chain that will achieve the
        /// pose.  If this computation fails, this function should return false.
        typedef std::function<bool(base::State*, const Eigen::Affine3d&)> EEInverseKinematicsFn;

        /// Representation of a pose constraint on the end effector of a kinematic
        /// chain.  This constraint is highly abstract and relies on callback
        /// functions to perform forward and inverse kinematics.
        class EndEffectorConstraint : public base::Constraint
        {
        public:
            /// \brief Constructor.  Takes (entire) state space and the location
            /// of the (sub)space that is being constrained.  Function pointers
            /// to forward and inverse kinematics routines are also required.
            EndEffectorConstraint(const base::StateSpacePtr& space,
                                  const base::StateSpace::SubstateLocation& loc,
                                  EEForwardKinematicsFn fk,
                                  EEInverseKinematicsFn ik) : base::Constraint(space)
            {
                loc_ = loc;
                fk_ = fk;
                ik_ = ik;

                // Identity pose, with no tolerances
                pos_ = Eigen::Vector3d(0, 0, 0);
                rpy_ = Eigen::Vector3d(0, 0, 0);
                posTol_ = Eigen::Vector3d(1e-6, 1e-6, 1e-6);
                rpyTol_ = Eigen::Vector3d(1e-6, 1e-6, 1e-6);
            }

            virtual ~EndEffectorConstraint()
            {
            }

            /// \brief Set the desired position of the end effector
            virtual void setPosition(double x, double y, double z)
            {
                pos_(0) = x;
                pos_(1) = y;
                pos_(2) = z;
            }

            /// \brief Set the tolerance around the desired position of the
            /// end effector to the given value, for each axis
            virtual void setPositionTolerance(double tol)
            {
                setPositionTolerance(tol, tol, tol);
            }

            virtual const Eigen::Vector3d& getPosition() const
            {
                return pos_;
            }

            virtual const Eigen::Vector3d& getOrientation() const
            {
                return rpy_;
            }

            virtual const Eigen::Vector3d& getPositionTol() const
            {
                return posTol_;
            }

            virtual const Eigen::Vector3d& getOrientationTol() const
            {
                return rpyTol_;
            }

            /// \brief Set the tolerance around the desired position of the
            /// end effector to the given values for each axis
            virtual void setPositionTolerance(double tolX, double tolY, double tolZ)
            {
                posTol_(0) = (tolX == 0.0 ? 1e-6 : tolX);
                posTol_(1) = (tolY == 0.0 ? 1e-6 : tolY);
                posTol_(2) = (tolZ == 0.0 ? 1e-6 : tolZ);
            }

            /// \brief Set the desired orientation of the end effector to the
            /// given Euler angles
            virtual void setOrientation(double roll, double pitch, double yaw)
            {
                rpy_(0) = roll;
                rpy_(1) = pitch;
                rpy_(2) = yaw;
            }

            /// \brief Set the tolerances allowed for the orientation about
            /// each axis to the given value (in radians)
            virtual void setOrientationTolerance(double tol)
            {
                setOrientationTolerance(tol, tol, tol);
            }

            /// \brief Set the tolerances allowed for the orientation about
            /// each axis to the given values (in radians)
            virtual void setOrientationTolerance(double tolR, double tolP, double tolY)
            {
                rpyTol_(0) = (tolR == 0.0 ? 1e-6 : tolR);
                rpyTol_(1) = (tolP == 0.0 ? 1e-6 : tolP);
                rpyTol_(2) = (tolY == 0.0 ? 1e-6 : tolY);
            }

            /// \brief Check whether this state satisfies the constraints
            virtual bool isSatisfied(const base::State* state) const
            {
                // Get a pointer to the portion of the state space
                // that this constraint refers to
                const base::State* substate = space_->getSubstateAtLocation(state, loc_);

                // Compute global reference frames using FK
                //std::vector<Eigen::Affine3d> frames;
                //fk_(substate, frames);
                Eigen::Affine3d frame;
                fk_(substate, frame);

                /*bool valid = true;
                for (size_t i = 0; i < 3; ++i)
                {
                    valid &= (fabs(pos_(i) - frames.back().translation()[i]) < posTol_(i));
                    if (i == 0)
                        std::cout << "x: ";
                    else if (i == 1)
                        std::cout << "y: ";
                    else std::cout << "z: ";
                    std::cout << frames.back().translation()[i] << "  Desired: " << pos_(i) << "   Error: " << fabs(pos_(i) - frames.back().translation()[i]) << " tolerance: " << posTol_(i) << std::endl;
                }

                // Checking orientation of last frame (end effector)
                Eigen::Vector3d rpy;
                rpy = frames.back().rotation().eulerAngles(0,1,2);
                for (size_t i = 0; i < 3; ++i)
                {
                    valid &= (fabs(rpy_(i) - rpy(i)) < rpyTol_(i));

                    if (i == 0)
                        std::cout << "R: ";
                    else if (i == 1)
                        std::cout << "P: ";
                    else std::cout << "Y: ";
                    std::cout << rpy(i) << "  Desired: " << rpy_(i) << "   Error: " << fabs(rpy_(i) - rpy(i)) << " tolerance: " << rpyTol_(i) << std::endl;
                }*/


                // Checking translation of last frame (end effector)
                bool valid = true;
                for (size_t i = 0; i < 3 && valid; ++i)
                    valid = (fabs(pos_(i) - frame.translation()[i]) < posTol_(i));

                // Checking orientation of last frame (end effector)
                Eigen::Vector3d rpy;
                rpy = frame.rotation().eulerAngles(0,1,2);
                for (size_t i = 0; i < 3 && valid; ++i)
                    valid = (fabs(rpy_(i) - rpy(i)) < rpyTol_(i));

                return valid;
            }

            /// Return a 6D vector containing the component-wise distances
            /// from the nominal pose
            virtual Eigen::VectorXd getDifference(const Eigen::Affine3d& frame)
            {
                Eigen::VectorXd diff(6);

                for (size_t i = 0; i < 3; ++i)
                {
                    diff(i) = pos_(i) - frame.translation()[i];

                    // Adjust for wiggle room
                    if (fabs(diff(i)) > posTol_(i))
                        diff(i) = fabs(diff(i) - posTol_(i));
                    else
                        diff(i) = 0.0;
                }

                Eigen::Vector3d rpy = frame.rotation().eulerAngles(0,1,2);
                for (size_t i = 0; i < 3; ++i)
                {
                    diff(i+3) = rpy_(i) - rpy(i);
                    // Adjust for wiggle room
                    if (fabs(diff(i+3)) > rpyTol_(i))
                        diff(i+3) = fabs(diff(i+3) - rpyTol_(i));
                    else
                        diff(i+3) = 0.0;
                }

                return diff;
            }

            /// Return a 3D vector containing the component-wise distances
            /// from the nominal pose in the z=0 plane
            virtual Eigen::VectorXd getDifference2D(const Eigen::Affine3d& frame)
            {
                Eigen::VectorXd diff(3);

                for (size_t i = 0; i < 2; ++i)
                {
                    diff(i) = pos_(i) - frame.translation()[i];

                    // Adjust for wiggle room
                    if (fabs(diff(i)) > posTol_(i))
                        diff(i) = fabs(diff(i) - posTol_(i));
                    else
                        diff(i) = 0.0;
                }

                Eigen::Vector3d rpy = frame.rotation().eulerAngles(0,1,2);
                diff(2) = rpy_(2) - rpy(2);
                // Adjust for wiggle room
                if (fabs(diff(2)) > rpyTol_(2))
                    diff(2) = fabs(diff(2) - rpyTol_(2));
                else
                    diff(2) = 0.0;

                return diff;
            }

            /// \brief Return the distance from satisfaction of a state
            virtual double distance(const base::State* state) const
            {
                // TODO: It is possible to take a distance.  See Berenson et.al. Task Space Region paper.
                return std::numeric_limits<double>::max();
            }

            /// \brief Sample a state given the constraints.  If a state cannot
            /// be sampled, this method will return false.
            virtual bool sample(base::State* state)
            {
                // Sampling a position within the tolerances
                Eigen::Vector3d offset;
                for(size_t i = 0; i < 3; ++i)
                    offset(i) = rng_.uniform01() * posTol_(i) * (rng_.uniform01() < 0.50 ? -1.0 : 1.0);

                Eigen::Translation3d posSample (pos_ + offset);

                // Sampling an orientation within the tolerances
                for(size_t i = 0; i < 3; ++i)
                    offset(i) = 2.0 * (rng_.uniform01() - 0.5) * rpyTol_(i);

                Eigen::Affine3d desired(Eigen::AngleAxisd(rpy_(0), Eigen::Vector3d::UnitX()) *
                                        Eigen::AngleAxisd(rpy_(1), Eigen::Vector3d::UnitY()) *
                                        Eigen::AngleAxisd(rpy_(2), Eigen::Vector3d::UnitZ()));

                Eigen::Affine3d delta(Eigen::AngleAxisd(offset(0), Eigen::Vector3d::UnitX()) *
                                      Eigen::AngleAxisd(offset(1), Eigen::Vector3d::UnitY()) *
                                      Eigen::AngleAxisd(offset(2), Eigen::Vector3d::UnitZ()));

                // Computing final sampled pose within tolerances
                Eigen::Affine3d sampledPose(posSample * desired * delta);

                // Sample the state by performing IK to reach the sampled pose
                return ik_(state, sampledPose);
            }

            /// \brief Project a state given the constraints.  If a valid
            /// projection cannot be found, this method will return false.
            virtual bool project(base::State* state)
            {
                // TODO: Should we first check for isSatisfied?
                if (!isSatisfied(state))
                    return sample(state);
                return true;
            }


        protected:
            base::StateSpace::SubstateLocation loc_;
            EEForwardKinematicsFn              fk_;
            EEInverseKinematicsFn              ik_;

            // The nominal pose for the end effector
            Eigen::Vector3d pos_;
            Eigen::Vector3d rpy_;

            // The tolerances allowed on the pose
            Eigen::Vector3d posTol_;
            Eigen::Vector3d rpyTol_;

            RNG rng_;
        };
    }
}

#endif
