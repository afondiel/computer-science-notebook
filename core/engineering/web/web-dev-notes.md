# Web Development - Notes

## Table of Contents
- [Introduction](#introduction)
  - [What's Web Development?](#whats-web-development)
  - [Applications](#applications)
- [Fundamentals](#fundamentals)
  - [How Web Development works?](#how-web-development-works)
    - [Web Development Architecture Pipeline](#web-development-architecture-pipeline)
- [Tools \& Frameworks](#tools--frameworks)
- [Hello World!](#hello-world)
- [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
- [References](#references)

## Introduction

Web development is the process of creating websites and web applications for the internet or intranet.

### What's Web Development?
Web development involves designing, coding, testing, and maintaining web pages and web applications using various technologies and languages. Web development can be divided into two main categories: front-end and back-end¹.

### Applications
Web development can be used to create various types of websites and web applications, such as:

- Static websites: Websites that display the same content to every visitor, such as personal portfolios, blogs, or landing pages.
- Dynamic websites: Websites that generate different content for each visitor, such as e-commerce, social media, or online learning platforms.
- Single-page applications (SPAs): Websites that load a single HTML page and dynamically update it as the user interacts with the web page, such as Gmail, Google Maps, or Facebook.
- Progressive web applications (PWAs): Websites that offer a native app-like experience on the web, such as Twitter, Pinterest, or Spotify.

## Fundamentals
To become a web developer, one needs to learn the following fundamental skills:

### How Web Development works?
Web development works by following a client-server model, where the client (browser) requests and displays web pages from the server (web server), which hosts and delivers the web pages and web applications.

#### Web Development Architecture Pipeline
The web development architecture pipeline consists of the following steps:

1. The client (browser) sends an HTTP request to the server (web server) for a web page or a web application.
2. The server (web server) receives the request and processes it using a back-end programming language, such as PHP, Python, or Node.js, and a database, such as MySQL, MongoDB, or PostgreSQL.
3. The server (web server) sends back an HTTP response to the client (browser) with the web page or the web application, which is composed of HTML, CSS, and JavaScript.
4. The client (browser) renders the web page or the web application using HTML, CSS, and JavaScript, and displays it to the user.
5. The client (browser) and the server (web server) can communicate with each other using AJAX (Asynchronous JavaScript and XML) or WebSockets, which allow data to be exchanged without reloading the web page.

## Tools & Frameworks
Web development can be enhanced and simplified by using various tools and frameworks, such as:

- **Web development tools:** Tools that help web developers create, test, debug, and optimize web pages and web applications, such as code editors, web browsers, web servers, web inspectors, and web performance tools.
- **Web development frameworks:** Frameworks that provide a set of pre-written code, libraries, and templates that web developers can use to build web pages and web applications faster and easier, such as Bootstrap, SASS, jQuery, React, and Django.

Front-end and back-end tools are software that help web developers create, test, and deploy websites and web applications. 
There are many different types of front-end and back-end tools, such as programming languages, frameworks, libraries, databases, web servers, and more. 

Here are some examples of popular front-end and back-end tools:

- **Front-end tools:**
  - HTML, CSS, and JavaScript: The core languages of the web, used to create the structure, style, and interactivity of web pages.
  - Bootstrap: A framework that provides ready-made components and templates for responsive web design.
  - React: A library that allows developers to create user interfaces using reusable components and state management.
  - SASS: A preprocessor that extends CSS with features such as variables, mixins, nesting, and more.
  - jQuery: A library that simplifies DOM manipulation, event handling, animation, and AJAX requests.
- **Back-end tools:**
  - PHP: A server-side scripting language that can be embedded in HTML and used to create dynamic web pages and web applications.
  - Python: A high-level, general-purpose programming language that supports multiple paradigms and has a rich set of libraries and frameworks for web development, such as Django and Flask.
  - Node.js: A runtime environment that allows developers to use JavaScript on the server-side and build scalable and fast web applications.
  - MySQL: A relational database management system that stores and retrieves data using SQL queries.
  - MongoDB: A non-relational database management system that stores and retrieves data using JSON-like documents.

## Hello World!
Here is a simple example of a web page that displays "Hello World!" using HTML, CSS, and JavaScript:

```html
<!DOCTYPE html>
<html>
<head>
  <style>
    h1 {
      color: blue;
      text-align: center;
    }
  </style>
</head>
<body>
  <h1 id="greeting"></h1>
  <script>
    document.getElementById("greeting").innerHTML = "Hello World!";
  </script>
</body>
</html>
```

## Lab: Zero to Hero Projects
To practice and improve your web development skills, you can try to create some of the following projects:

- A personal portfolio website that showcases your skills, projects, and contact information.
- A blog website that allows you to write and publish posts, and allows visitors to comment and share them.
- A todo list web application that allows you to add, edit, delete, and mark tasks as done, and stores them in the local storage or a database.
- A weather web application that displays the current weather and forecast for a given location, using a weather API.
- A calculator web application that performs basic arithmetic operations, using HTML, CSS, and JavaScript.

## References

- (1) [Web Development - W3Schools](https://www.w3schools.com/whatis/).
- (2) [Learn web development | MDN - MDN Web Docs](https://developer.mozilla.org/en-US/docs/Learn).
- (3) [Learn web development | MDN - MDN Web Docs](https://developer.mozilla.org/en-US/docs/Learn).
- (4) [What Does a Web Developer Do (and How Do I Become One)? - Coursera](https://www.coursera.org/articles/web-developer).

- [A Handpicked Selection of 60+ Best Web Development Tools - Radixweb.com](https://radixweb.com/blog/web-development-tools)

Online Courses & Lectures
- [Full Stack Web Development for Beginners (Full Course on HTML, CSS, JavaScript, Node.js, MongoDB) - freeCodeCamp.org](https://www.youtube.com/watch?v=nu_pCVPKzTk&list=PLQpme5qo9tFDdmv1J6KQLVAhBu2Ne9sYh&index=3)
- [Frontend Web Development Bootcamp Course (JavaScript, HTML, CSS) -  Zach Gollwitzer/freeCodeCamp](https://www.youtube.com/watch?v=zJSY8tbf_ys&list=PLQpme5qo9tFDdmv1J6KQLVAhBu2Ne9sYh&index=4)
- [Use ChatGPT to Code a Full Stack App – Full Course - Judy/freeCodeCamp](https://www.youtube.com/watch?v=GizsSo-EevA&list=PLQpme5qo9tFDdmv1J6KQLVAhBu2Ne9sYh&index=5)



