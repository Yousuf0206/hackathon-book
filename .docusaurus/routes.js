import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/hackathon-book/__docusaurus/debug',
    component: ComponentCreator('/hackathon-book/__docusaurus/debug', 'c29'),
    exact: true
  },
  {
    path: '/hackathon-book/__docusaurus/debug/config',
    component: ComponentCreator('/hackathon-book/__docusaurus/debug/config', 'ead'),
    exact: true
  },
  {
    path: '/hackathon-book/__docusaurus/debug/content',
    component: ComponentCreator('/hackathon-book/__docusaurus/debug/content', '336'),
    exact: true
  },
  {
    path: '/hackathon-book/__docusaurus/debug/globalData',
    component: ComponentCreator('/hackathon-book/__docusaurus/debug/globalData', 'bb6'),
    exact: true
  },
  {
    path: '/hackathon-book/__docusaurus/debug/metadata',
    component: ComponentCreator('/hackathon-book/__docusaurus/debug/metadata', '644'),
    exact: true
  },
  {
    path: '/hackathon-book/__docusaurus/debug/registry',
    component: ComponentCreator('/hackathon-book/__docusaurus/debug/registry', 'e50'),
    exact: true
  },
  {
    path: '/hackathon-book/__docusaurus/debug/routes',
    component: ComponentCreator('/hackathon-book/__docusaurus/debug/routes', 'e1b'),
    exact: true
  },
  {
    path: '/hackathon-book/docs',
    component: ComponentCreator('/hackathon-book/docs', 'd4a'),
    routes: [
      {
        path: '/hackathon-book/docs',
        component: ComponentCreator('/hackathon-book/docs', '5f3'),
        routes: [
          {
            path: '/hackathon-book/docs',
            component: ComponentCreator('/hackathon-book/docs', '9fd'),
            routes: [
              {
                path: '/hackathon-book/docs/appendix/',
                component: ComponentCreator('/hackathon-book/docs/appendix/', '311'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/appendix/hardware-setup/',
                component: ComponentCreator('/hackathon-book/docs/appendix/hardware-setup/', '4b8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/appendix/lab-requirements/',
                component: ComponentCreator('/hackathon-book/docs/appendix/lab-requirements/', '142'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/appendix/tools-installation/',
                component: ComponentCreator('/hackathon-book/docs/appendix/tools-installation/', '235'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/appendix/tools-installation/gazebo',
                component: ComponentCreator('/hackathon-book/docs/appendix/tools-installation/gazebo', '03f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/appendix/tools-installation/isaac',
                component: ComponentCreator('/hackathon-book/docs/appendix/tools-installation/isaac', 'd94'),
                exact: true
              },
              {
                path: '/hackathon-book/docs/appendix/tools-installation/ros2',
                component: ComponentCreator('/hackathon-book/docs/appendix/tools-installation/ros2', '4d3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/appendix/tools-installation/unity',
                component: ComponentCreator('/hackathon-book/docs/appendix/tools-installation/unity', 'cdd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/bibliography/references',
                component: ComponentCreator('/hackathon-book/docs/bibliography/references', '15f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/chapter-template',
                component: ComponentCreator('/hackathon-book/docs/chapter-template', '2cc'),
                exact: true
              },
              {
                path: '/hackathon-book/docs/modules/module1-ros2/chapter1-introduction/',
                component: ComponentCreator('/hackathon-book/docs/modules/module1-ros2/chapter1-introduction/', '2c6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module1-ros2/chapter2-architecture/',
                component: ComponentCreator('/hackathon-book/docs/modules/module1-ros2/chapter2-architecture/', '6c3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module1-ros2/chapter3-packages/',
                component: ComponentCreator('/hackathon-book/docs/modules/module1-ros2/chapter3-packages/', 'd6f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module1-ros2/chapter4-urdf/',
                component: ComponentCreator('/hackathon-book/docs/modules/module1-ros2/chapter4-urdf/', '233'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module1-ros2/chapter5-control/',
                component: ComponentCreator('/hackathon-book/docs/modules/module1-ros2/chapter5-control/', '772'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module1-ros2/preface',
                component: ComponentCreator('/hackathon-book/docs/modules/module1-ros2/preface', '28c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module2-digital-twin/chapter1-digital-twins/',
                component: ComponentCreator('/hackathon-book/docs/modules/module2-digital-twin/chapter1-digital-twins/', '8cb'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module2-digital-twin/chapter2-gazebo/',
                component: ComponentCreator('/hackathon-book/docs/modules/module2-digital-twin/chapter2-gazebo/', '97d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module2-digital-twin/chapter3-sensor-simulation/',
                component: ComponentCreator('/hackathon-book/docs/modules/module2-digital-twin/chapter3-sensor-simulation/', 'b98'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module2-digital-twin/chapter4-unity/',
                component: ComponentCreator('/hackathon-book/docs/modules/module2-digital-twin/chapter4-unity/', 'fec'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module2-digital-twin/chapter5-environment-building/',
                component: ComponentCreator('/hackathon-book/docs/modules/module2-digital-twin/chapter5-environment-building/', '6fa'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module2-digital-twin/preface',
                component: ComponentCreator('/hackathon-book/docs/modules/module2-digital-twin/preface', '20f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module3-ai-brain/chapter1-isaac-ecosystem/',
                component: ComponentCreator('/hackathon-book/docs/modules/module3-ai-brain/chapter1-isaac-ecosystem/', '696'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module3-ai-brain/chapter2-synthetic-data/',
                component: ComponentCreator('/hackathon-book/docs/modules/module3-ai-brain/chapter2-synthetic-data/', '8db'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module3-ai-brain/chapter3-perception-pipelines/',
                component: ComponentCreator('/hackathon-book/docs/modules/module3-ai-brain/chapter3-perception-pipelines/', '48c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module3-ai-brain/chapter4-navigation/',
                component: ComponentCreator('/hackathon-book/docs/modules/module3-ai-brain/chapter4-navigation/', 'd7d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module3-ai-brain/chapter5-reinforcement-learning/',
                component: ComponentCreator('/hackathon-book/docs/modules/module3-ai-brain/chapter5-reinforcement-learning/', 'afe'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module3-ai-brain/preface',
                component: ComponentCreator('/hackathon-book/docs/modules/module3-ai-brain/preface', '5f4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module4-vla/chapter1-vla-introduction/',
                component: ComponentCreator('/hackathon-book/docs/modules/module4-vla/chapter1-vla-introduction/', '980'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module4-vla/chapter2-voice-to-action/',
                component: ComponentCreator('/hackathon-book/docs/modules/module4-vla/chapter2-voice-to-action/', '391'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module4-vla/chapter3-cognitive-planning/',
                component: ComponentCreator('/hackathon-book/docs/modules/module4-vla/chapter3-cognitive-planning/', '42b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module4-vla/chapter4-multi-modal-perception/',
                component: ComponentCreator('/hackathon-book/docs/modules/module4-vla/chapter4-multi-modal-perception/', '97d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module4-vla/chapter5-capstone-project/',
                component: ComponentCreator('/hackathon-book/docs/modules/module4-vla/chapter5-capstone-project/', 'caf'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/modules/module4-vla/preface',
                component: ComponentCreator('/hackathon-book/docs/modules/module4-vla/preface', '5bf'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/hackathon-book/docs/preface/',
                component: ComponentCreator('/hackathon-book/docs/preface/', 'bec'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/hackathon-book/',
    component: ComponentCreator('/hackathon-book/', 'ee1'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
