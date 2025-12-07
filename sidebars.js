/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Preface',
      items: ['preface/README'],
      link: {
        type: 'doc',
        id: 'preface/README',
      },
    },
    {
      type: 'category',
      label: 'Module 1: ROS 2',
      items: [
        'modules/module1-ros2/preface',
        'modules/module1-ros2/chapter1-introduction/README',
        'modules/module1-ros2/chapter2-architecture/README',
        'modules/module1-ros2/chapter3-packages/README',
        'modules/module1-ros2/chapter4-urdf/README',
        'modules/module1-ros2/chapter5-control/README',
      ],
      link: {
        type: 'doc',
        id: 'modules/module1-ros2/preface',
      },
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twins',
      items: [
        'modules/module2-digital-twin/preface',
        'modules/module2-digital-twin/chapter1-digital-twins/README',
        'modules/module2-digital-twin/chapter2-gazebo/README',
        'modules/module2-digital-twin/chapter3-sensor-simulation/README',
        'modules/module2-digital-twin/chapter4-unity/README',
        'modules/module2-digital-twin/chapter5-environment-building/README',
      ],
      link: {
        type: 'doc',
        id: 'modules/module2-digital-twin/preface',
      },
    },
    {
      type: 'category',
      label: 'Module 3: AI Brain',
      items: [
        'modules/module3-ai-brain/preface',
        'modules/module3-ai-brain/chapter1-isaac-ecosystem/README',
        'modules/module3-ai-brain/chapter2-synthetic-data/README',
        'modules/module3-ai-brain/chapter3-perception-pipelines/README',
        'modules/module3-ai-brain/chapter4-navigation/README',
        'modules/module3-ai-brain/chapter5-reinforcement-learning/README',
      ],
      link: {
        type: 'doc',
        id: 'modules/module3-ai-brain/preface',
      },
    },
    {
      type: 'category',
      label: 'Module 4: VLA',
      items: [
        'modules/module4-vla/preface',
        'modules/module4-vla/chapter1-vla-introduction/README',
        'modules/module4-vla/chapter2-voice-to-action/README',
        'modules/module4-vla/chapter3-cognitive-planning/README',
        'modules/module4-vla/chapter4-multi-modal-perception/README',
        'modules/module4-vla/chapter5-capstone-project/README',
      ],
      link: {
        type: 'doc',
        id: 'modules/module4-vla/preface',
      },
    },
    {
      type: 'category',
      label: 'Appendix',
      items: [
        'appendix/README',
        'appendix/hardware-setup/README',
        'appendix/lab-requirements/README',
        'appendix/tools-installation/README',
        'appendix/tools-installation/ros2',
        'appendix/tools-installation/gazebo',
        'appendix/tools-installation/unity',
      ],
    },
    {
      type: 'doc',
      id: 'bibliography/references',
    },
  ],
};

module.exports = sidebars;