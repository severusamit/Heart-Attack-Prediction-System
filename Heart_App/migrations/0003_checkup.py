# Generated by Django 4.2 on 2023-04-07 08:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Heart_App', '0002_admindetails'),
    ]

    operations = [
        migrations.CreateModel(
            name='Checkup',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Date', models.CharField(default=None, max_length=100, null=True)),
                ('Oraganised', models.CharField(default=None, max_length=100, null=True)),
                ('Place', models.CharField(default=None, max_length=100, null=True)),
            ],
            options={
                'db_table': 'Checkup',
            },
        ),
    ]
