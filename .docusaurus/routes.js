import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/docs',
    component: ComponentCreator('/docs', '2cf'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '17d'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '9d5'),
            routes: [
              {
                path: '/docs/appendix/',
                component: ComponentCreator('/docs/appendix/', 'e3d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/appendix/hardware-setup/',
                component: ComponentCreator('/docs/appendix/hardware-setup/', '48b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/appendix/lab-requirements/',
                component: ComponentCreator('/docs/appendix/lab-requirements/', '732'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/appendix/tools-installation/',
                component: ComponentCreator('/docs/appendix/tools-installation/', '14c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/appendix/tools-installation/gazebo',
                component: ComponentCreator('/docs/appendix/tools-installation/gazebo', 'b03'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/appendix/tools-installation/isaac',
                component: ComponentCreator('/docs/appendix/tools-installation/isaac', '0f4'),
                exact: true
              },
              {
                path: '/docs/appendix/tools-installation/ros2',
                component: ComponentCreator('/docs/appendix/tools-installation/ros2', 'c87'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/appendix/tools-installation/unity',
                component: ComponentCreator('/docs/appendix/tools-installation/unity', '4d0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/bibliography/references',
                component: ComponentCreator('/docs/bibliography/references', '244'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/chapter-template',
                component: ComponentCreator('/docs/chapter-template', '7f6'),
                exact: true
              },
              {
                path: '/docs/modules/module1-ros2/chapter1-introduction/',
                component: ComponentCreator('/docs/modules/module1-ros2/chapter1-introduction/', '81b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module1-ros2/chapter2-architecture/',
                component: ComponentCreator('/docs/modules/module1-ros2/chapter2-architecture/', '037'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module1-ros2/chapter3-packages/',
                component: ComponentCreator('/docs/modules/module1-ros2/chapter3-packages/', 'f08'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module1-ros2/chapter4-urdf/',
                component: ComponentCreator('/docs/modules/module1-ros2/chapter4-urdf/', 'a5c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module1-ros2/chapter5-control/',
                component: ComponentCreator('/docs/modules/module1-ros2/chapter5-control/', '4ec'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module1-ros2/preface',
                component: ComponentCreator('/docs/modules/module1-ros2/preface', '6a8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module2-digital-twin/chapter1-digital-twins/',
                component: ComponentCreator('/docs/modules/module2-digital-twin/chapter1-digital-twins/', '30a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module2-digital-twin/chapter2-gazebo/',
                component: ComponentCreator('/docs/modules/module2-digital-twin/chapter2-gazebo/', '12d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module2-digital-twin/chapter3-sensor-simulation/',
                component: ComponentCreator('/docs/modules/module2-digital-twin/chapter3-sensor-simulation/', 'f09'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module2-digital-twin/chapter4-unity/',
                component: ComponentCreator('/docs/modules/module2-digital-twin/chapter4-unity/', '286'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module2-digital-twin/chapter5-environment-building/',
                component: ComponentCreator('/docs/modules/module2-digital-twin/chapter5-environment-building/', 'b9b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module2-digital-twin/preface',
                component: ComponentCreator('/docs/modules/module2-digital-twin/preface', 'a55'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module3-ai-brain/chapter1-isaac-ecosystem/',
                component: ComponentCreator('/docs/modules/module3-ai-brain/chapter1-isaac-ecosystem/', '6c0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module3-ai-brain/chapter2-synthetic-data/',
                component: ComponentCreator('/docs/modules/module3-ai-brain/chapter2-synthetic-data/', '271'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module3-ai-brain/chapter3-perception-pipelines/',
                component: ComponentCreator('/docs/modules/module3-ai-brain/chapter3-perception-pipelines/', 'e99'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module3-ai-brain/chapter4-navigation/',
                component: ComponentCreator('/docs/modules/module3-ai-brain/chapter4-navigation/', '672'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module3-ai-brain/chapter5-reinforcement-learning/',
                component: ComponentCreator('/docs/modules/module3-ai-brain/chapter5-reinforcement-learning/', '514'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module3-ai-brain/preface',
                component: ComponentCreator('/docs/modules/module3-ai-brain/preface', '88e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module4-vla/chapter1-vla-introduction/',
                component: ComponentCreator('/docs/modules/module4-vla/chapter1-vla-introduction/', '082'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module4-vla/chapter2-voice-to-action/',
                component: ComponentCreator('/docs/modules/module4-vla/chapter2-voice-to-action/', '6cf'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module4-vla/chapter3-cognitive-planning/',
                component: ComponentCreator('/docs/modules/module4-vla/chapter3-cognitive-planning/', '4e4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module4-vla/chapter4-multi-modal-perception/',
                component: ComponentCreator('/docs/modules/module4-vla/chapter4-multi-modal-perception/', 'c97'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module4-vla/chapter5-capstone-project/',
                component: ComponentCreator('/docs/modules/module4-vla/chapter5-capstone-project/', '925'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module4-vla/preface',
                component: ComponentCreator('/docs/modules/module4-vla/preface', '7c2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/preface/',
                component: ComponentCreator('/docs/preface/', '2c8'),
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
    path: '/',
    component: ComponentCreator('/', '2e1'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
